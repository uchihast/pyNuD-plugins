"""
SpotAnalysis
------------
Standalone spot-count analysis module. For each frame, the module fits a
sum of 2D Gaussian functions with 2 or 3 peaks, selects the model by AIC/BIC,
applies band-pass preprocessing and centroid tracking, and automatically
evaluates the S/N of the third peak.
"""

from __future__ import annotations

import logging
import os
import csv
import glob
import re
import json
import html
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit, linear_sum_assignment
from skimage import feature, filters, morphology, segmentation, measure
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, Circle

import globalvals as gv
from fileio import LoadFrame, InitializeAryDataFallback

logger = logging.getLogger(__name__)

# Plugin display name shown in the Plugin menu
PLUGIN_NAME = "Spot Analysis"


HELP_CSS = """
<style>
body {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  font-size: 13px;
  line-height: 1.55;
  color: #20242a;
}
h2 {
  font-size: 18px;
  margin: 0 0 10px 0;
}
h3 {
  font-size: 15px;
  margin: 16px 0 6px 0;
}
p {
  margin: 6px 0 10px 0;
}
ol, ul {
  margin-top: 6px;
  margin-bottom: 10px;
  padding-left: 24px;
}
li {
  margin: 5px 0;
}
table {
  border-collapse: collapse;
  margin: 8px 0 12px 0;
  width: 100%;
}
th, td {
  border: 1px solid #d8dde6;
  padding: 6px 8px;
  vertical-align: top;
}
th {
  background: #eef2f7;
}
.note {
  background: #fff7df;
  border: 1px solid #efd37a;
  border-radius: 6px;
  padding: 8px 10px;
  margin: 10px 0;
}
.good {
  background: #eaf7ef;
  border: 1px solid #9ed3ad;
  border-radius: 6px;
  padding: 8px 10px;
  margin: 10px 0;
}
.term {
  font-weight: 600;
  color: #111827;
}
code {
  background: #f0f2f5;
  padding: 1px 4px;
  border-radius: 3px;
}
</style>
"""


HELP_TABS_EN = [
    (
        "Quick Start",
        """
        <h2>Recommended first workflow</h2>
        <ol>
          <li>Open AFM image data in pyNuD and show the file/frame you want to analyze.</li>
          <li>Open <span class="term">Spot Analysis</span>. The full-image overlay window opens together with the control panel.</li>
          <li>In the upper AFM image of the overlay window, drag the area where spots should be searched to create an ROI.</li>
          <li>Click <span class="term">Run Analysis</span> to detect spots and select the Gaussian peak model for the current frame.</li>
          <li>If needed, adjust <span class="term">Detection</span>, <span class="term">Initialization</span>, and <span class="term">Result Filters</span>.</li>
          <li>For a frame series, enable the <span class="term">Run All Frames</span> checkbox, then click the <span class="term">Run All Frames</span> button.</li>
          <li>After checking the result, use <span class="term">Export CSV</span> to save spot coordinates, heights, S/N values, and related metadata.</li>
        </ol>
        <div class="note">
          <b>Core idea:</b><br>
          The ROI is the search area. Detection creates initial spot candidates.
          Fitting refines those candidates with Gaussian models. The final peak count is selected by AIC or BIC within the requested peak range.
        </div>
        """,
    ),
    (
        "Views",
        """
        <h2>What each view shows</h2>
        <table>
          <tr><th>Area</th><th>Purpose</th><th>What to check</th></tr>
          <tr>
            <td><span class="term">Spot Analysis</span> control panel</td>
            <td>Sets ROI shape, detection, initialization, Gaussian fitting, result filters, display, and CSV output.</td>
            <td>Work from ROI / Basic through Preprocessing, Detection, Initialization, Fit, Result Filters, and Display / Recording.</td>
          </tr>
          <tr>
            <td>Full overlay window, upper image</td>
            <td>Shows the full AFM image, ROI outline, and final spot positions.</td>
            <td>Check whether the ROI covers the intended area and whether final spots sit on real bright features.</td>
          </tr>
          <tr>
            <td>Lower-left ROI image</td>
            <td>Shows the preprocessed ROI image.</td>
            <td>Check whether median/open preprocessing removed noise without removing the real spots.</td>
          </tr>
          <tr>
            <td>Lower-right detection image</td>
            <td>Shows the LoG/DoG detection image or the selected detection image.</td>
            <td>Check whether seed candidates align with bright spots and whether noise is being over-detected.</td>
          </tr>
          <tr>
            <td>Result text</td>
            <td>Reports selected peak count, AIC/BIC, S/N values, and spot coordinates.</td>
            <td>Check whether the peak count and S/N values are reasonable and whether any filters removed candidates.</td>
          </tr>
        </table>
        <div class="good">
          When a result looks wrong, compare the final positions in the upper view with the detection image below.
          This separates detection problems from fitting or filtering problems.
        </div>
        """,
    ),
    (
        "ROI and Frames",
        """
        <h2>ROI, frame navigation, and batch analysis</h2>
        <h3>Creating an ROI</h3>
        <ul>
          <li>Select Rectangle or Ellipse (Circle) in <span class="term">ROI Shape</span>.</li>
          <li>Drag on the upper AFM image in the full overlay window to save an ROI for the current frame.</li>
          <li><span class="term">ROI Margin</span> excludes peaks near the ROI boundary. Use it when edge peaks are unstable.</li>
        </ul>
        <h3>Frame navigation</h3>
        <ul>
          <li>Use <span class="term">◀</span> / <span class="term">▶</span> to move to the previous or next frame.</li>
          <li>When <span class="term">Auto Analyze on Frame Change</span> is enabled, frames with an ROI are analyzed automatically.</li>
          <li>If a later frame has no ROI, the most recent earlier ROI is propagated.</li>
        </ul>
        <h3>All-frame analysis</h3>
        <ul>
          <li>The <span class="term">Run All Frames</span> button is enabled only when its checkbox is turned on.</li>
          <li>Before running all frames, validate ROI, detection, and filters on representative frames.</li>
          <li>If you manually edited spots, avoid unintended reanalysis before exporting the CSV.</li>
        </ul>
        """,
    ),
    (
        "Detection and Fit",
        """
        <h2>How to tune parameters</h2>
        <table>
          <tr><th>Panel</th><th>Purpose</th><th>Practical tuning</th></tr>
          <tr>
            <td>Preprocessing</td>
            <td>Reduces noise or small protrusions inside the ROI.</td>
            <td><span class="term">Median</span> helps point noise. <span class="term">Open</span> removes small structures.</td>
          </tr>
          <tr>
            <td>Detection</td>
            <td>Chooses the image used to create seed candidates.</td>
            <td>Start with <span class="term">LoG</span>, then tune sigma to the expected spot size.</td>
          </tr>
          <tr>
            <td>Initialization</td>
            <td>Creates initial peak positions for Gaussian fitting.</td>
            <td><span class="term">Watershed</span> is robust for complex images, <span class="term">Blob DoH</span> is good for round spots, and <span class="term">Peak</span> is simple local-maximum detection.</td>
          </tr>
          <tr>
            <td>Fit</td>
            <td>Refines candidate positions with Gaussian models.</td>
            <td>If fitting is unstable, tune <span class="term">Initial Sigma</span> and sigma lower/upper bounds to the spot diameter.</td>
          </tr>
          <tr>
            <td>Result Filters</td>
            <td>Removes low-S/N, weak, boundary, or spatial outlier peaks.</td>
            <td>Raising <span class="term">S/N Threshold</span>, <span class="term">precheck</span>, or <span class="term">Min Amplitude</span> makes filtering stricter.</td>
          </tr>
        </table>
        <div class="note">
          AIC tends to accept additional peaks more easily. BIC is more conservative.
          If noise is selected as a third peak, check BIC, S/N Threshold, and precheck settings.
        </div>
        """,
    ),
    (
        "Editing and CSV",
        """
        <h2>Manual editing, centroid correction, and CSV</h2>
        <h3>Spot editing</h3>
        <ul>
          <li><span class="term">Ctrl/Cmd + drag</span>: move an existing spot.</li>
          <li><span class="term">Shift + click</span>: add a spot.</li>
          <li><span class="term">Alt(Option) + click</span>: delete a spot.</li>
        </ul>
        <h3>Move spots to the spot-radius centroid</h3>
        <ul>
          <li><span class="term">Spot Radius</span> controls both the display circle and the circular area used for mean height.</li>
          <li><span class="term">Move Spots to Radius Centroid</span> moves spots in the current frame to the intensity centroid inside each spot-radius circle.</li>
          <li>Enable <span class="term">Auto Apply</span> to apply this centroid correction immediately after analysis.</li>
        </ul>
        <h3>CSV</h3>
        <ul>
          <li><span class="term">Height Value</span> selects whether the primary CSV height is the spot-position height or the spot-radius mean.</li>
          <li><span class="term">Export CSV</span> saves per-frame spot coordinates, peak heights, S/N values, and related metadata.</li>
          <li><span class="term">Load CSV -> Restore</span> restores ROI and in-progress spot display state from saved CSV files.</li>
        </ul>
        """,
    ),
    (
        "Troubleshooting",
        """
        <h2>Common issues</h2>
        <table>
          <tr><th>Symptom</th><th>What to check</th></tr>
          <tr>
            <td>Run Analysis is disabled</td>
            <td>Make sure a file is selected and an ROI exists for the current frame.</td>
          </tr>
          <tr>
            <td>Cannot draw an ROI</td>
            <td>Check that the full overlay window is open and drag on the upper AFM image with the left mouse button.</td>
          </tr>
          <tr>
            <td>Too many spots are detected</td>
            <td>Try BIC, raise S/N Threshold, raise precheck K(MAD), increase peak spacing, or increase detection sigma.</td>
          </tr>
          <tr>
            <td>Real spots are missed</td>
            <td>Lower S/N Threshold or precheck, tune LoG sigma to the spot size, and check whether Median/Open is too strong.</td>
          </tr>
          <tr>
            <td>Fitted positions drift away</td>
            <td>Tune Initial Sigma and sigma bounds, disable fitting to use seed positions only, or correct spots manually.</td>
          </tr>
          <tr>
            <td>Restored CSV results change unexpectedly</td>
            <td>Check whether auto analysis or parameter changes re-ran analysis and overwrote restored spots.</td>
          </tr>
        </table>
        <div class="good">
          Start by tuning one representative frame, then apply the validated settings to all frames.
        </div>
        """,
    ),
]

HELP_TABS_JA = [
    (
        "最短手順",
        """
        <h2>まずはこの順番で解析する</h2>
        <ol>
          <li>pyNuD本体でAFM画像を開き、解析したいファイルとフレームを表示します。</li>
          <li><span class="term">Spot Analysis</span> を開くと、全画像表示ウィンドウも開きます。</li>
          <li>全画像表示ウィンドウの上段AFM画像で、スポットを探したい範囲をドラッグしてROIを作ります。</li>
          <li><span class="term">Run Analysis</span> を押して、現在フレームのスポット検出とガウスモデル判定を行います。</li>
          <li>結果を見て、必要なら <span class="term">Detection</span>、<span class="term">Initialization</span>、<span class="term">Result Filters</span> を調整します。</li>
          <li>複数フレームに適用する場合は <span class="term">Run All Frames</span> をONにしてから、<span class="term">Run All Frames</span> ボタンを押します。</li>
          <li>確認後、<span class="term">Export CSV</span> でスポット座標・高さ・S/Nなどを保存します。</li>
        </ol>
        <div class="note">
          <b>基本の考え方:</b><br>
          ROIは「探す範囲」、検出は「初期候補を見つける方法」、フィットは「候補をガウスモデルで最適化する工程」です。
          最終的なピーク数は、指定したピーク数範囲の中からAIC/BICで選ばれます。
        </div>
        """,
    ),
    (
        "画面の見方",
        """
        <h2>ウィンドウと表示の役割</h2>
        <table>
          <tr><th>場所</th><th>役割</th><th>確認すること</th></tr>
          <tr>
            <td><span class="term">Spot Analysis</span> 操作パネル</td>
            <td>ROI形状、検出方法、初期候補、ガウスフィット、結果フィルタ、CSV出力を設定します。</td>
            <td>ROI / Basic、Preprocessing、Detection、Initialization、Fit、Result Filters、Display / Recordingを順に調整します。</td>
          </tr>
          <tr>
            <td>全画像表示ウィンドウ上段</td>
            <td>AFM全体画像とROI、最終スポット位置を表示します。</td>
            <td>ROIが目的の範囲を囲んでいるか、スポットが実際の輝点に乗っているか確認します。</td>
          </tr>
          <tr>
            <td>下段左のROI画像</td>
            <td>前処理後のROI画像です。</td>
            <td>median/open後に、解析したいスポットが消えていないか確認します。</td>
          </tr>
          <tr>
            <td>下段右の検出画像</td>
            <td>LoG/DoGなど検出用画像です。</td>
            <td>初期候補が輝点に対応しているか、ノイズを拾いすぎていないか確認します。</td>
          </tr>
          <tr>
            <td>結果テキスト</td>
            <td>採用ピーク数、AIC/BIC、S/N、ピーク座標などを表示します。</td>
            <td>ピーク数とS/Nが妥当か、除外された理由がないか確認します。</td>
          </tr>
        </table>
        <div class="good">
          迷ったら、全画像表示の上段で最終位置、下段で検出画像を見比べてください。
          「検出が悪い」のか「フィット後にずれている」のかを分けて判断できます。
        </div>
        """,
    ),
    (
        "ROIと解析",
        """
        <h2>ROI、フレーム移動、全フレーム解析</h2>
        <h3>ROIの作り方</h3>
        <ul>
          <li><span class="term">ROI Shape</span> で Rectangle または Ellipse (Circle) を選びます。</li>
          <li>全画像表示ウィンドウの上段AFM画像をドラッグすると、ROIが現在フレームに保存されます。</li>
          <li><span class="term">ROI Margin</span> は、ROI境界近くのピークを除外する距離です。境界で不安定なピークを避けたい時に使います。</li>
        </ul>
        <h3>フレーム移動</h3>
        <ul>
          <li><span class="term">◀</span> / <span class="term">▶</span> で前後フレームへ移動します。</li>
          <li><span class="term">Auto Analyze on Frame Change</span> がONの場合、ROIがあるフレームでは自動的に解析します。</li>
          <li>ROIがないフレームでは、直近過去フレームのROIを引き継ぎます。</li>
        </ul>
        <h3>全フレーム解析</h3>
        <ul>
          <li>誤操作防止のため、チェックボックスの <span class="term">Run All Frames</span> をONにした時だけボタンが有効になります。</li>
          <li>全フレームに同じ設定を適用する前に、代表フレームでROI、検出、フィルタを確認してください。</li>
          <li>解析結果を手動修正している場合、CSV保存前に意図せず再解析で上書きしないよう注意してください。</li>
        </ul>
        """,
    ),
    (
        "検出とフィット",
        """
        <h2>パラメータ調整の考え方</h2>
        <table>
          <tr><th>パネル</th><th>主な目的</th><th>調整の目安</th></tr>
          <tr>
            <td>Preprocessing</td>
            <td>ROI内のノイズや小さな突起を抑えます。</td>
            <td><span class="term">Median</span> は点ノイズに有効です。<span class="term">Open</span> は小さな構造を削ります。</td>
          </tr>
          <tr>
            <td>Detection</td>
            <td>ピーク候補を作る画像を決めます。</td>
            <td>通常は <span class="term">LoG</span> から始めます。スポットサイズに合わせてσを変えます。</td>
          </tr>
          <tr>
            <td>Initialization</td>
            <td>ガウスフィットへ渡す初期ピーク位置を作ります。</td>
            <td><span class="term">Watershed</span> は複雑な画像向け、<span class="term">Blob DoH</span> は円形スポット向け、<span class="term">Peak</span> は単純な局所最大です。</td>
          </tr>
          <tr>
            <td>Fit</td>
            <td>候補位置をガウスモデルで最適化します。</td>
            <td>フィットが不安定なら、<span class="term">Initial Sigma</span> とσ下限/上限をスポット径に合わせます。</td>
          </tr>
          <tr>
            <td>Result Filters</td>
            <td>低S/N、弱いピーク、境界近く、外れ値を除外します。</td>
            <td><span class="term">S/N Threshold</span>、<span class="term">precheck</span>、<span class="term">Min Amplitude</span> を上げると厳しくなります。</td>
          </tr>
        </table>
        <div class="note">
          AICはピーク数を増やしやすく、BICはより保守的です。
          ノイズを3つ目のピークとして拾う場合は、BIC、S/N Threshold、precheckを見直してください。
        </div>
        """,
    ),
    (
        "手動編集とCSV",
        """
        <h2>手動修正、重心補正、CSV保存</h2>
        <h3>Spot編集</h3>
        <ul>
          <li><span class="term">Ctrl/Cmd + ドラッグ</span>: 既存スポットを移動します。</li>
          <li><span class="term">Shift + クリック</span>: スポットを追加します。</li>
          <li><span class="term">Alt(Option) + クリック</span>: スポットを削除します。</li>
        </ul>
        <h3>Spot径の重心へ移動</h3>
        <ul>
          <li><span class="term">Spot Radius</span> は、表示円の大きさと、円内平均高さの計算範囲に使われます。</li>
          <li><span class="term">Move Spots to Radius Centroid</span> は、現在フレームの各スポットを円内の強度重心へ移動します。</li>
          <li><span class="term">Auto Apply</span> をONにすると、解析後に重心補正を自動で行います。</li>
        </ul>
        <h3>CSV</h3>
        <ul>
          <li><span class="term">Height Value</span> で、CSVの主高さをSpot位置またはSpot径内平均から選びます。</li>
          <li><span class="term">Export CSV</span> は、フレームごとのスポット座標、ピーク高さ、S/Nなどを保存します。</li>
          <li><span class="term">Load CSV -> Restore</span> は、保存済みCSVからROIやスポット表示状態を復元します。</li>
        </ul>
        """,
    ),
    (
        "困ったとき",
        """
        <h2>よくあるトラブル</h2>
        <table>
          <tr><th>症状</th><th>確認すること</th></tr>
          <tr>
            <td>Run Analysisを実行できない</td>
            <td>ファイルが選択され、現在フレームにROIが作られているか確認してください。</td>
          </tr>
          <tr>
            <td>ROIを描けない</td>
            <td>全画像表示ウィンドウが開いているか、上段AFM画像上で左ドラッグしているか確認してください。</td>
          </tr>
          <tr>
            <td>スポット数が多すぎる</td>
            <td>BICに変更する、S/N Thresholdを上げる、precheck K(MAD)を上げる、ピーク間隔を広げる、検出σを大きくする、の順で確認します。</td>
          </tr>
          <tr>
            <td>スポットを拾わない</td>
            <td>S/N Thresholdやprecheckを下げる、LoG σをスポット径に合わせる、Median/Openが強すぎないか確認します。</td>
          </tr>
          <tr>
            <td>フィット後に位置がずれる</td>
            <td>Initial Sigmaとσ範囲を見直し、必要ならフィットをOFFにして初期位置だけを使うか、手動編集で補正します。</td>
          </tr>
          <tr>
            <td>CSV復元後に結果が変わる</td>
            <td>自動解析やパラメータ変更による再解析で上書きされていないか確認してください。</td>
          </tr>
        </table>
        <div class="good">
          まず代表フレーム1枚でROIと検出条件を固めてから、全フレーム解析へ進むと調整が速くなります。
        </div>
        """,
    ),
]


@dataclass
class PeakStat:
    amplitude: float
    x: float
    y: float
    sigma: float
    snr: float


@dataclass
class ModelSelectionResult:
    n_peaks: int
    popt: np.ndarray
    rss: float
    loglike: float
    aic: float
    bic: float
    residual_std: float
    fit_applied: bool
    init_peaks: List[PeakStat]
    peaks: List[PeakStat]


@dataclass
class FrameAnalysis:
    best_n_peaks: int
    criterion: str
    noise_sigma: float
    snr_threshold: float
    models: Dict[int, ModelSelectionResult]
    roi: np.ndarray  # ROI image for visualization
    origin: Tuple[int, int]  # ROI origin in the source image (x0, y0)
    roi_mask: Optional[np.ndarray] = None  # ROI mask such as ellipse mask
    seed_spots: Optional[List[Dict[str, float]]] = None  # Seed points from detection stage in absolute coordinates, used for display.


class SpotAnalysis:
    """
    Class for AIC/BIC comparison between 2- and 3-Gaussian models and
    S/N evaluation.

    Example:
        sa = SpotAnalysis(roi_size=48, snr_threshold=2.0)
        result = sa.analyze_frame(frame)  # frame: 2D numpy array
        print(result.best_n_peaks, result.models[2].aic, result.models[3].aic)
    """

    def __init__(
        self,
        roi_size: int = 48,
        bandpass_low_sigma: float = 0.6,
        bandpass_high_sigma: float = 3.0,
        detection_mode: str = "LoG",
        log_sigma: float = 1.6,
        median_enabled: bool = False,
        median_size: int = 3,
        open_enabled: bool = False,
        open_radius: int = 1,
        init_mode: str = "peak",
        blob_min_sigma: float = 3.0,
        blob_max_sigma: float = 20.0,
        blob_num_sigma: int = 10,
        blob_threshold_rel: float = 0.05,
        blob_overlap: float = 0.5,
        blob_doh_min_sigma: float = 3.0,
        blob_doh_max_sigma: float = 20.0,
        blob_doh_num_sigma: int = 10,
        blob_doh_threshold_rel: float = 0.01,
        watershed_h_rel: float = 0.05,
        watershed_adaptive_h: bool = True,
        peak_min_distance: int = 5,
        localmax_threshold_rel: float = 0.05,
        localmax_threshold_snr: float = 0.0,
        precheck_radius_px: int = 2,
        precheck_kmad: float = 1.0,
        multiscale_enabled: bool = False,
        multiscale_sigmas: Optional[Sequence[float]] = None,
        dbscan_enabled: bool = False,
        dbscan_eps: float = 5.0,
        dbscan_min_samples: int = 1,
        snap_enabled: bool = False,
        snap_radius: int = 2,
        snap_refit_enabled: bool = False,
        refit_max_shift_px: int = 3,
        fit_enabled: bool = True,
        sigma_bounds: Tuple[float, float] = (0.1, 4.0),
        initial_sigma: float = 0.2,
        snr_threshold: float = 1.0,
        max_iterations: int = 8000,
        margin: int = 5,
        min_amplitude: float = 0.0,
        min_sigma_result: float = 0.0,
    ) -> None:
        self.roi_size = max(8, int(roi_size))
        self.bandpass_low_sigma = float(bandpass_low_sigma)
        self.bandpass_high_sigma = float(bandpass_high_sigma)
        self.detection_mode = str(detection_mode).strip().lower()  # "dog" / "log" / "pre"
        self.log_sigma = float(log_sigma)
        self.median_enabled = bool(median_enabled)
        self.median_size = int(median_size)
        self.open_enabled = bool(open_enabled)
        self.open_radius = int(open_radius)
        self.init_mode = str(init_mode).strip().lower()  # "peak" / "peak_subpixel" / "blob" / "blob_doh" / "watershed" / "multiscale"
        self.blob_min_sigma = float(blob_min_sigma)
        self.blob_max_sigma = float(blob_max_sigma)
        self.blob_num_sigma = max(1, int(blob_num_sigma))
        self.blob_threshold_rel = float(blob_threshold_rel)
        self.blob_overlap = float(blob_overlap)
        self.blob_doh_min_sigma = float(blob_doh_min_sigma)
        self.blob_doh_max_sigma = float(blob_doh_max_sigma)
        self.blob_doh_num_sigma = max(1, int(blob_doh_num_sigma))
        self.blob_doh_threshold_rel = float(blob_doh_threshold_rel)
        self.watershed_h_rel = float(watershed_h_rel)
        self.watershed_adaptive_h = bool(watershed_adaptive_h)
        self.peak_min_distance = max(1, int(peak_min_distance))
        self.localmax_threshold_rel = float(localmax_threshold_rel)
        self.localmax_threshold_snr = float(localmax_threshold_snr)
        self.precheck_radius_px = max(0, int(precheck_radius_px))
        self.precheck_kmad = float(precheck_kmad)
        self.multiscale_enabled = bool(multiscale_enabled)
        self.multiscale_sigmas = list(multiscale_sigmas) if multiscale_sigmas else [1.0, 1.6, 2.5, 4.0]
        self.dbscan_enabled = bool(dbscan_enabled)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = max(1, int(dbscan_min_samples))
        self.snap_enabled = bool(snap_enabled)
        self.snap_radius = max(0, int(snap_radius))
        self.snap_refit_enabled = bool(snap_refit_enabled)
        self.refit_max_shift_px = max(0, int(refit_max_shift_px))
        self.fit_enabled = bool(fit_enabled)
        self.sigma_bounds = (max(1e-3, sigma_bounds[0]), max(sigma_bounds[0] + 1e-3, sigma_bounds[1]))
        self.initial_sigma = float(initial_sigma)
        self.snr_threshold = float(snr_threshold)
        self.max_iterations = max(1000, int(max_iterations))
        self.margin = int(margin)
        self.min_amplitude = float(min_amplitude)
        self.min_sigma_result = float(min_sigma_result)

    def detection_label(self) -> str:
        mode = (self.detection_mode or "dog").lower()
        if mode == "log":
            return f"LoG (sigma={self.log_sigma:g})"
        if mode in ("pre", "preprocessed"):
            return "Pre (median/open)"
        return f"DoG (sigma={self.bandpass_low_sigma:g}-{self.bandpass_high_sigma:g})"

    def _apply_roi_preprocess(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self.median_enabled and int(self.median_size) > 1:
            k = int(self.median_size)
            if k % 2 == 0:
                k += 1
            # square footprint is intuitive for UI "kernel size"
            out = filters.median(out, footprint=morphology.square(k))
        if self.open_enabled and int(self.open_radius) > 0:
            r = int(self.open_radius)
            out = morphology.opening(out, morphology.disk(r))
        return out

    def _apply_detection(self, img: np.ndarray) -> np.ndarray:
        mode = (self.detection_mode or "dog").lower()
        if mode == "log":
            sigma = max(float(self.log_sigma), 1e-6)
            # Use negative sign so bright blobs become positive peaks
            return -ndimage.gaussian_laplace(img, sigma=sigma)
        if mode in ("pre", "preprocessed"):
            return img
        # DoG
        # NOTE: DoG creates ring-like patterns around spots, which can lead to
        # false peak detection at valleys instead of true peaks.
        # We mitigate this by validating peaks against the preprocessed image.
        if self.bandpass_high_sigma <= self.bandpass_low_sigma:
            return img
        return filters.difference_of_gaussians(img, self.bandpass_low_sigma, self.bandpass_high_sigma)

    def _roi_mask_with_margin(
        self, roi_mask: Optional[np.ndarray], margin_px: int, shape: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Build an ROI mask that excludes margin from the ROI boundary.

        - If roi_mask is provided (ellipse etc.), it is eroded by margin_px.
        - Independently, we also exclude a rectangular border of width margin_px for ROI crops.
        """
        try:
            m = int(margin_px)
        except Exception:
            m = 0
        m = max(0, m)

        if roi_mask is not None:
            try:
                mask = roi_mask.astype(bool)
            except Exception:
                mask = None
        else:
            mask = None

        if mask is not None and m > 0:
            try:
                mask = morphology.erosion(mask, morphology.disk(m))
            except Exception:
                pass

        if shape is None:
            if mask is None:
                return None
            h, w = mask.shape
        else:
            h, w = shape
            if mask is None:
                try:
                    mask = np.ones((int(h), int(w)), dtype=bool)
                except Exception:
                    return None
            else:
                try:
                    if mask.shape != (int(h), int(w)):
                        # If shape mismatch, ignore mask (caller will fall back to border-only)
                        mask = np.ones((int(h), int(w)), dtype=bool)
                except Exception:
                    mask = np.ones((int(h), int(w)), dtype=bool)

        if m > 0 and h > 2 * m and w > 2 * m:
            try:
                mask = mask.copy()
                mask[:m, :] = False
                mask[-m:, :] = False
                mask[:, :m] = False
                mask[:, -m:] = False
            except Exception:
                pass
        return mask

    def _snap_to_local_max_pixel(
        self,
        img: np.ndarray,
        y: int,
        x: int,
        radius: int,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, bool]:
        """
        Snap (y,x) to the maximum-intensity pixel within a (2r+1)x(2r+1) window.
        Does NOT reject; only moves the coordinate when a better pixel exists.
        Returns (new_y, new_x, moved).
        """
        frame = np.asarray(img, dtype=np.float64)
        if frame.ndim != 2:
            return int(y), int(x), False
        h, w = frame.shape
        iy = int(np.clip(int(y), 0, h - 1))
        ix = int(np.clip(int(x), 0, w - 1))
        try:
            r = max(0, int(radius))
        except Exception:
            r = 0
        if r <= 0:
            return iy, ix, False

        y0 = max(0, iy - r)
        y1 = min(h, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(w, ix + r + 1)
        if y0 >= y1 or x0 >= x1:
            return iy, ix, False

        best_y, best_x = iy, ix
        try:
            best_v = float(frame[iy, ix])
        except Exception:
            best_v = float("-inf")

        for ny in range(y0, y1):
            for nx in range(x0, x1):
                if roi_mask is not None:
                    try:
                        if not bool(roi_mask[ny, nx]):
                            continue
                    except Exception:
                        pass
                try:
                    v = float(frame[ny, nx])
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                if v > best_v:
                    best_v = v
                    best_y, best_x = ny, nx

        moved = (best_y != iy) or (best_x != ix)
        return int(best_y), int(best_x), bool(moved)

    def _pipeline(self, img: np.ndarray) -> np.ndarray:
        return self._apply_detection(self._apply_roi_preprocess(img))

    def _detect_local_maxima(
        self,
        img: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        threshold_rel: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        allow_more: int = 4,
        return_meta: bool = False,
    ) -> np.ndarray:
        """
        Detect local maxima (pixel-grid) on the given image.
        Returns coords as ndarray of shape (k, 2) in (row=y, col=x).

        Notes:
        - This runs on the detection image (LoG/DoG after preprocess), not on the raw image.
        - threshold_rel is relative to max(img); if None, uses self.localmax_threshold_rel.
        """
        frame = np.asarray(img, dtype=np.float64)
        if frame.ndim != 2:
            return np.zeros((0, 2), dtype=np.int64)
        h, w = frame.shape
        if h == 0 or w == 0:
            return np.zeros((0, 2), dtype=np.int64)

        thr_rel = self.localmax_threshold_rel if threshold_rel is None else float(threshold_rel)
        try:
            thr_rel = float(thr_rel)
        except Exception:
            thr_rel = 0.0
        thr_rel = min(max(thr_rel, 0.0), 1.0)
        v_max = float(np.nanmax(frame)) if np.size(frame) else float("nan")
        threshold_abs = None
        if np.isfinite(v_max) and v_max > 0 and thr_rel > 0:
            threshold_abs = thr_rel * v_max
        try:
            snr_thr = float(getattr(self, "localmax_threshold_snr", 0.0))
        except Exception:
            snr_thr = 0.0
        if noise_sigma is not None and snr_thr > 0:
            try:
                snr_abs = snr_thr * float(noise_sigma)
            except Exception:
                snr_abs = 0.0
            if np.isfinite(snr_abs) and snr_abs > 0:
                threshold_abs = snr_abs if threshold_abs is None else max(threshold_abs, snr_abs)

        num_peaks = int(max(1, n_peaks))
        num_peaks = int(max(num_peaks, num_peaks * int(max(1, allow_more))))

        # Determine appropriate border exclusion based on image size
        # Exclude at least 3 pixels from border to avoid DoG/LoG artifacts
        border_exclude = max(3, int(min(h, w) * 0.05))  # 5% of smaller dimension, minimum 3px
        try:
            m0 = max(0, int(getattr(self, "margin", 0)))
        except Exception:
            m0 = 0

        used_mask = None
        used_m = 0
        coords = np.zeros((0, 2), dtype=np.int64)

        # Stepwise margin relaxation: prefer excluding ROI margin, but guarantee we can return peaks.
        for m in range(int(m0), -1, -1):
            use_border = int(max(border_exclude, m))
            used_m = int(m)
            used_mask = self._roi_mask_with_margin(roi_mask, use_border, shape=(h, w))
            labels = None
            if used_mask is not None:
                try:
                    labels = used_mask.astype(np.int32)
                except Exception:
                    labels = None
            try:
                coords_try = feature.peak_local_max(
                    frame,
                    min_distance=int(self.peak_min_distance),
                    num_peaks=num_peaks,
                    exclude_border=use_border,
                    threshold_abs=threshold_abs,
                    labels=labels,
                )
                if coords_try is None:
                    coords_try = np.zeros((0, 2), dtype=np.int64)
            except Exception:
                coords_try = np.zeros((0, 2), dtype=np.int64)
            coords = np.asarray(coords_try, dtype=np.int64)
            if coords.shape[0] >= int(n_peaks):
                break

        if return_meta:
            return coords, used_mask, used_m
        return coords

    def _top_pixels_by_value_nms(
        self,
        img: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        min_distance: Optional[int] = None,
    ) -> np.ndarray:
        """
        Pick peak seed coordinates by descending pixel value with greedy non-maximum suppression.

        This is used for init_mode=peak / peak_subpixel to make the initializer match the
        visualized detection image (LoG/DoG) more directly than peak_local_max.

        Rules:
        - sort pixels by value (descending) on the detection image
        - keep only pixels inside ROI-mask + ROI-margin (and rectangle border exclusion)
        - enforce a minimum separation (min_distance px) between selected seeds (greedy NMS)

        Returns coords as ndarray of shape (k,2) in (row=y, col=x).
        """
        frame = np.asarray(img, dtype=np.float64)
        if frame.ndim != 2:
            return np.zeros((0, 2), dtype=np.int64)
        h, w = frame.shape
        if h == 0 or w == 0:
            return np.zeros((0, 2), dtype=np.int64)

        try:
            n = int(n_peaks)
        except Exception:
            n = 0
        if n <= 0:
            return np.zeros((0, 2), dtype=np.int64)

        try:
            r = int(self.peak_min_distance if min_distance is None else min_distance)
        except Exception:
            r = int(getattr(self, "peak_min_distance", 2) or 0)
        r = max(0, int(r))

        # Determine border exclusion similarly to _detect_local_maxima (avoid edge artifacts)
        border_exclude = max(3, int(min(h, w) * 0.05))
        try:
            m0 = max(0, int(getattr(self, "margin", 0)))
        except Exception:
            m0 = 0

        accepted: List[Tuple[int, int]] = []

        # Like _detect_local_maxima, relax margin stepwise only if we cannot collect enough seeds.
        for m in range(int(m0), -1, -1):
            use_border = int(max(border_exclude, m))
            allowed = self._roi_mask_with_margin(roi_mask, use_border, shape=(h, w))
            if allowed is None:
                allowed = np.ones((int(h), int(w)), dtype=bool)

            # Mask out disallowed/invalid pixels with -inf so they sort to the end.
            masked = np.where(allowed & np.isfinite(frame), frame, -np.inf)
            flat = masked.ravel()
            if flat.size == 0:
                continue

            order = np.argsort(flat)[::-1]
            for idx in order:
                if len(accepted) >= int(n):
                    break
                v = float(flat[int(idx)])
                if not np.isfinite(v) or v == float("-inf"):
                    # Remaining candidates are all invalid/disallowed.
                    break
                y = int(int(idx) // int(w))
                x = int(int(idx) % int(w))

                if r > 0 and accepted:
                    ok = True
                    for ay, ax in accepted:
                        # Window-based NMS: suppress within (2r+1)x(2r+1) neighborhood
                        # (Chebyshev distance <= r).
                        dy = abs(int(ay) - int(y))
                        dx = abs(int(ax) - int(x))
                        if dx <= r and dy <= r:
                            ok = False
                            break
                    if not ok:
                        continue

                accepted.append((y, x))

            if len(accepted) >= int(n):
                break

        if not accepted:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(accepted[: int(n)], dtype=np.int64)

    def _refine_subpixel_quadratic(
        self,
        img: np.ndarray,
        x0: int,
        y0: int,
    ) -> Tuple[float, float]:
        """
        Subpixel refinement around (x0,y0) using 3x3 quadratic approximation.
        Returns (x,y) floats in image coordinates.
        """
        frame = np.asarray(img, dtype=np.float64)
        h, w = frame.shape
        if x0 <= 0 or y0 <= 0 or x0 >= w - 1 or y0 >= h - 1:
            return float(x0), float(y0)

        p = frame[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2]
        if p.shape != (3, 3) or not np.all(np.isfinite(p)):
            return float(x0), float(y0)

        # Central differences for gradient
        gx = 0.5 * (p[1, 2] - p[1, 0])
        gy = 0.5 * (p[2, 1] - p[0, 1])

        # Hessian (second derivatives)
        gxx = p[1, 2] - 2.0 * p[1, 1] + p[1, 0]
        gyy = p[2, 1] - 2.0 * p[1, 1] + p[0, 1]
        gxy = 0.25 * (p[2, 2] - p[2, 0] - p[0, 2] + p[0, 0])

        det = gxx * gyy - gxy * gxy
        if not np.isfinite(det) or abs(det) < 1e-12:
            return float(x0), float(y0)

        # Newton step: offset = -H^{-1} * grad
        dx = (-gyy * gx + gxy * gy) / det
        dy = (gxy * gx - gxx * gy) / det

        if not np.isfinite(dx) or not np.isfinite(dy):
            return float(x0), float(y0)

        # Keep within a reasonable neighborhood to avoid jumping
        dx = float(np.clip(dx, -0.5, 0.5))
        dy = float(np.clip(dy, -0.5, 0.5))
        return float(x0) + dx, float(y0) + dy

    def _snap_peaks_to_local_maxima(
        self,
        peaks: List[PeakStat],
        det_img: np.ndarray,
        origin: Tuple[int, int],
        roi_mask: Optional[np.ndarray] = None,
    ) -> List[PeakStat]:
        """
        Snap peak coordinates to the nearest strong local maximum in det_img within snap_radius.
        - det_img: ROI-local detection image (after preprocess + LoG/DoG).
        - origin: ROI origin in absolute coords.
        """
        if not peaks or self.snap_radius <= 0:
            return peaks
        img = np.asarray(det_img, dtype=np.float64)
        if img.ndim != 2:
            return peaks
        h, w = img.shape
        x0, y0 = int(origin[0]), int(origin[1])

        allowed = None
        if roi_mask is not None:
            try:
                allowed = roi_mask.astype(bool)
                if allowed.shape != (h, w):
                    allowed = None
            except Exception:
                allowed = None

        r = int(self.snap_radius)
        used: set[Tuple[int, int]] = set()
        snapped: List[PeakStat] = []

        # Prefer snapping higher-SNR peaks first to reduce collisions.
        order = sorted(range(len(peaks)), key=lambda i: float(getattr(peaks[i], "snr", 0.0)), reverse=True)
        for i in order:
            pk = peaks[i]
            try:
                lx = float(pk.x) - float(x0)
                ly = float(pk.y) - float(y0)
            except Exception:
                snapped.append(pk)
                continue
            ix = int(round(lx))
            iy = int(round(ly))
            ix = int(np.clip(ix, 0, w - 1))
            iy = int(np.clip(iy, 0, h - 1))

            x_min = max(0, ix - r)
            x_max = min(w - 1, ix + r)
            y_min = max(0, iy - r)
            y_max = min(h - 1, iy + r)
            window = img[y_min : y_max + 1, x_min : x_max + 1]
            if window.size == 0 or not np.any(np.isfinite(window)):
                snapped.append(pk)
                continue

            # Candidate maxima in this window (sorted by response).
            flat = window.ravel()
            finite_mask = np.isfinite(flat)
            if not np.any(finite_mask):
                snapped.append(pk)
                continue
            idxs = np.argsort(flat[finite_mask])[::-1]
            finite_positions = np.flatnonzero(finite_mask)

            chosen = None
            for j in idxs:
                pos = int(finite_positions[int(j)])
                wy, wx = divmod(pos, window.shape[1])
                cx = x_min + int(wx)
                cy = y_min + int(wy)
                if (cx, cy) in used:
                    continue
                if allowed is not None and not bool(allowed[cy, cx]):
                    continue
                chosen = (cx, cy)
                break

            if chosen is None:
                snapped.append(pk)
                continue
            used.add(chosen)

            cx, cy = chosen
            # Optional subpixel refinement around snapped integer maximum.
            rx, ry = self._refine_subpixel_quadratic(img, cx, cy)
            rx = float(np.clip(rx, 0.0, float(w - 1)))
            ry = float(np.clip(ry, 0.0, float(h - 1)))

            snapped.append(
                PeakStat(
                    amplitude=pk.amplitude,
                    x=float(x0) + rx,
                    y=float(y0) + ry,
                    sigma=pk.sigma,
                    snr=pk.snr,
                )
            )

        # Restore original order
        inv = {idx: out for idx, out in zip(order, snapped)}
        return [inv[i] for i in range(len(peaks))]

    def _enforce_min_peak_distance(
        self, peaks: List[PeakStat], return_rejections: bool = False
    ) -> Any:
        """
        Enforce minimum distance between peaks in absolute coordinates.

        Policy:
        - Use self.peak_min_distance (px).
        - Keep higher-SNR peaks first (tie-breaker: higher amplitude).
        - Greedy selection; drops peaks closer than min_distance to any kept peak.

        NOTE: This is a post-process on the *final* peak list. It can reduce the
        number of peaks below the UI min_peaks when the user prioritizes distance.
        """
        rejected_map: Dict[int, Dict[str, Any]] = {}

        def _add_reject(pk: PeakStat, reason: str) -> None:
            if not return_rejections:
                return
            k = id(pk)
            ent = rejected_map.get(k)
            if ent is None:
                idx = getattr(pk, "_prefilter_index", None)
                ent = {"peak": pk, "reasons": [], "prefilter_index": idx}
                rejected_map[k] = ent
            elif ent.get("prefilter_index") is None:
                ent["prefilter_index"] = getattr(pk, "_prefilter_index", None)
            ent["reasons"].append(reason)

        try:
            d_min = float(self.peak_min_distance)
        except Exception:
            return (peaks, []) if return_rejections else peaks
        if not peaks or not np.isfinite(d_min) or d_min <= 0:
            return (peaks, []) if return_rejections else peaks

        d2_min = float(d_min) ** 2
        # Sort by (snr desc, amplitude desc)
        order = sorted(
            peaks,
            key=lambda p: (float(getattr(p, "snr", 0.0)), float(getattr(p, "amplitude", 0.0))),
            reverse=True,
        )

        kept: List[PeakStat] = []
        for pk in order:
            try:
                x = float(pk.x)
                y = float(pk.y)
            except Exception:
                continue
            ok = True
            for kp in kept:
                try:
                    dx = x - float(kp.x)
                    dy = y - float(kp.y)
                except Exception:
                    continue
                if dx * dx + dy * dy < d2_min:
                    dist = float(np.hypot(dx, dy))
                    _add_reject(
                        pk,
                        f"Peak spacing(min={d_min:.1f}px, dist={dist:.2f}px)",
                    )
                    ok = False
                    break
            if ok:
                kept.append(pk)

        # Preserve stable labeling order by x/y (optional) is not desired; keep original relative order.
        kept_set = set(id(p) for p in kept)
        kept_in_original = [p for p in peaks if id(p) in kept_set]
        if return_rejections:
            return kept_in_original, list(rejected_map.values())
        return kept_in_original

    def _ensure_min_peak_count(
        self,
        kept_peaks: List[PeakStat],
        original_peaks: List[PeakStat],
        min_keep: int,
        rejected_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Guarantee that at least ``min_keep`` peaks remain in the final output.

        If post-filters reduce the count below the requested minimum, restore
        peaks from the selected model's original peak list in descending
        priority (higher S/N, then higher amplitude). Restored peaks are
        inserted back in the original model order.
        """
        try:
            target = int(min_keep)
        except Exception:
            target = 0
        target = max(0, int(target))

        kept = list(kept_peaks or [])
        original = list(original_peaks or [])
        if target <= 0 or len(kept) >= target or not original:
            if rejected_infos is None:
                return kept
            return kept, list(rejected_infos)

        kept_ids = {id(pk) for pk in kept}
        candidates = [pk for pk in original if id(pk) not in kept_ids]
        candidates.sort(
            key=lambda p: (float(getattr(p, "snr", 0.0)), float(getattr(p, "amplitude", 0.0))),
            reverse=True,
        )

        needed = max(0, int(target - len(kept)))
        restored = candidates[:needed]
        if not restored:
            if rejected_infos is None:
                return kept
            return kept, list(rejected_infos)

        restored_ids = {id(pk) for pk in restored}
        merged_ids = kept_ids | restored_ids
        merged = [pk for pk in original if id(pk) in merged_ids]

        if rejected_infos is None:
            return merged

        kept_rejections: List[Dict[str, Any]] = []
        for ent in list(rejected_infos):
            try:
                pk = ent.get("peak")
            except Exception:
                pk = None
            if pk is None or id(pk) in restored_ids:
                continue
            kept_rejections.append(ent)
        return merged, kept_rejections

    def _force_min_peak_count_on_model(self, model: ModelSelectionResult, min_keep: int) -> None:
        """
        Hard safety net for the final selected model.

        If the model still has fewer than ``min_keep`` peaks at the end of
        post-processing, restore peaks from the pre-filter list (preferred) or
        from init_peaks as a fallback.
        """
        try:
            target = max(0, int(min_keep))
        except Exception:
            target = 0
        if target <= 0 or model is None:
            return

        current = list(getattr(model, "peaks", []) or [])
        if len(current) >= target:
            return

        original = list(getattr(model, "prefilter_peaks", []) or [])
        if not original:
            original = list(getattr(model, "init_peaks", []) or [])
        if not original:
            return

        try:
            restored_peaks, kept_rejections = self._ensure_min_peak_count(
                current,
                original,
                target,
                rejected_infos=list(getattr(model, "excluded_infos", []) or []),
            )
        except Exception:
            restored_peaks = current
            kept_rejections = list(getattr(model, "excluded_infos", []) or [])

        model.peaks = list(restored_peaks or [])
        try:
            model.excluded_infos = list(kept_rejections or [])
        except Exception:
            pass

    def _filter_peaks(
        self,
        peaks: List[PeakStat],
        roi_origin: Tuple[int, int],
        roi_shape: Tuple[int, int],
        roi_pre: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        return_rejections: bool = False,
    ) -> Any:
        """
        Filter detected peaks.
        1. Lower-bound filters (sigma, amplitude)
        2. S/N filter (snr_threshold; output filter)
        3. Local-maximum validation on the original ROI image (roi_pre; median+MAD / local maximum)
        4. ROI boundary filter (margin, roi_mask)
        """
        rejected_map: Dict[int, Dict[str, Any]] = {}

        def _add_reject(pk: PeakStat, reason: str) -> None:
            if not return_rejections:
                return
            k = id(pk)
            ent = rejected_map.get(k)
            if ent is None:
                idx = getattr(pk, "_prefilter_index", None)
                ent = {"peak": pk, "reasons": [], "prefilter_index": idx}
                rejected_map[k] = ent
            elif ent.get("prefilter_index") is None:
                ent["prefilter_index"] = getattr(pk, "_prefilter_index", None)
            ent["reasons"].append(reason)

        filtered: List[PeakStat] = []

        # 1. Lower-bound filter
        for pk in peaks:
            try:
                sigma_v = float(pk.sigma)
            except Exception:
                sigma_v = float("nan")
            try:
                amp_v = float(pk.amplitude)
            except Exception:
                amp_v = float("nan")

            if np.isfinite(sigma_v) and sigma_v < float(self.min_sigma_result):
                _add_reject(pk, f"Minσ: sigma={sigma_v:.3g} < thr={float(self.min_sigma_result):.3g}")
                continue
            if np.isfinite(amp_v) and amp_v < float(self.min_amplitude):
                _add_reject(pk, f"Minimum Amplitude: amp={amp_v:.3g} < thr={float(self.min_amplitude):.3g}")
                continue
            filtered.append(pk)

        # 2. S/N filter for output. If all peaks are below threshold, keep them instead of dropping all.
        if filtered and self.snr_threshold is not None:
            thr = float(self.snr_threshold)
            if thr > 0:
                snr_pass = [pk for pk in filtered if float(pk.snr) >= thr]
                if snr_pass:
                    for pk in filtered:
                        try:
                            snr_v = float(pk.snr)
                        except Exception:
                            snr_v = float("nan")
                        if np.isfinite(snr_v) and snr_v < thr:
                            _add_reject(pk, f"S/N threshold: snr={snr_v:.3g} < thr={thr:.3g}")
                    filtered = snr_pass
                else:
                    # If no peak exceeds the threshold, keep all peaks to avoid an empty-looking result.
                    filtered = list(filtered)

        # 3. Local-peak validation on the original image (roi_pre)
        if filtered and roi_pre is not None:
            try:
                r = int(self.precheck_radius_px)
            except Exception:
                r = 0
            try:
                kmad = float(self.precheck_kmad)
            except Exception:
                kmad = 0.0

            if r > 0 or kmad > 0:
                h, w = roi_shape
                x0, y0 = roi_origin
                mask = None
                if roi_mask is not None:
                    try:
                        mask = roi_mask.astype(bool)
                        if mask.shape != (h, w):
                            mask = None
                    except Exception:
                        mask = None
                kept_tmp: List[PeakStat] = []
                for pk in filtered:
                    try:
                        lx = float(pk.x) - float(x0)
                        ly = float(pk.y) - float(y0)
                    except Exception:
                        _add_reject(pk, "precheck: coordinate transform failed")
                        continue
                    ix0 = int(round(lx))
                    iy0 = int(round(ly))
                    ix = ix0
                    iy = iy0
                    if ix < 0 or iy < 0 or ix >= w or iy >= h:
                        _add_reject(pk, "precheck: outside ROI")
                        continue

                    # Snap to the maximum pixel within precheck radius (instead of rejecting).
                    moved = False
                    if r > 0:
                        # Try strict mask first; if nothing finite, relax mask.
                        roi_mask_for_snap = mask
                        iy_s, ix_s, moved = self._snap_to_local_max_pixel(
                            roi_pre, iy, ix, radius=int(r), roi_mask=roi_mask_for_snap
                        )
                        ix, iy = int(ix_s), int(iy_s)
                        if moved:
                            try:
                                pk.x = float(x0 + ix)
                                pk.y = float(y0 + iy)
                            except Exception:
                                pass

                    x_min = max(0, ix - r)
                    x_max = min(w - 1, ix + r)
                    y_min = max(0, iy - r)
                    y_max = min(h - 1, iy + r)
                    win = roi_pre[y_min : y_max + 1, x_min : x_max + 1]
                    if win.size == 0:
                        _add_reject(pk, "precheck: empty window")
                        continue

                    if mask is not None:
                        try:
                            win_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
                            win = np.where(win_mask, win, np.nan)
                        except Exception:
                            pass

                    # If no finite reference, do not reject.
                    if not np.any(np.isfinite(win)):
                        kept_tmp.append(pk)
                        continue

                    try:
                        v0 = float(roi_pre[iy, ix])
                    except Exception:
                        _add_reject(pk, "precheck: failed to get v0")
                        continue
                    if not np.isfinite(v0):
                        _add_reject(pk, "precheck: v0 is non-finite")
                        continue

                    ok = True
                    if kmad > 0:
                        med = float(np.nanmedian(win))
                        mad = float(np.nanmedian(np.abs(win - med)))
                        if not np.isfinite(mad) or mad <= 0:
                            mad = float(np.nanstd(win))
                        if not np.isfinite(mad) or mad <= 0:
                            mad = 0.0
                        thr_kmad = med + kmad * mad
                        if v0 < thr_kmad:
                            ok = False
                            snap_str = ""
                            if moved:
                                snap_str = f"precheck snap: ({ix0},{iy0})->({ix},{iy}); "
                            _add_reject(
                                pk,
                                snap_str
                                + f"precheck K(MAD): v0={v0:.3g} < med+K*MAD={thr_kmad:.3g} (med={med:.3g}, MAD={mad:.3g}, K={kmad:.2g})",
                            )

                    if ok:
                        kept_tmp.append(pk)

                filtered = kept_tmp

        # 4. ROI boundary filter
        h, w = roi_shape
        x0, y0 = roi_origin

        # If an ROI mask is provided (ellipse etc.), always enforce peaks to be inside it.
        # When margin > 0, additionally erode the mask to exclude boundary-adjacent peaks.
        if roi_mask is not None:
            try:
                base_allowed = roi_mask.astype(bool)
                allowed = base_allowed
                if self.margin > 0:
                    allowed = morphology.erosion(allowed, morphology.disk(int(self.margin)))
            except Exception:
                allowed = None

            if allowed is not None and allowed.shape == (h, w):
                temp: List[PeakStat] = []
                for pk in filtered:
                    lx = float(pk.x) - float(x0)
                    ly = float(pk.y) - float(y0)
                    ix = int(np.floor(lx))
                    iy = int(np.floor(ly))
                    if 0 <= ix < w and 0 <= iy < h and bool(allowed[iy, ix]):
                        temp.append(pk)
                    else:
                        try:
                            if 0 <= ix < w and 0 <= iy < h and "base_allowed" in locals() and base_allowed.shape == (h, w):
                                if not bool(base_allowed[iy, ix]):
                                    _add_reject(pk, "outside ROI (mask)")
                                else:
                                    _add_reject(pk, f"ROI margin(margin={int(self.margin)}px)")
                            else:
                                _add_reject(pk, "outside ROI")
                        except Exception:
                            _add_reject(pk, "outside ROI")
                filtered = temp

        # Rectangle ROI: apply simple margin-to-border exclusion.
        elif self.margin > 0:
            min_x = x0 + self.margin
            max_x = x0 + w - self.margin
            min_y = y0 + self.margin
            max_y = y0 + h - self.margin

            temp2: List[PeakStat] = []
            for pk in filtered:
                if min_x <= pk.x < max_x and min_y <= pk.y < max_y:
                    temp2.append(pk)
                else:
                    try:
                        # distance to nearest border in px (rough)
                        dx = min(float(pk.x) - float(x0), float(x0 + w) - float(pk.x))
                        dy = min(float(pk.y) - float(y0), float(y0 + h) - float(pk.y))
                        d_border = min(dx, dy)
                        _add_reject(pk, f"ROI margin(margin={int(self.margin)}px, border_dist~{d_border:.2f}px)")
                    except Exception:
                        _add_reject(pk, f"ROI margin(margin={int(self.margin)}px)")
            filtered = temp2

        # 4. DBSCAN spatial outlier filtering (optional)
        if self.dbscan_enabled and filtered:
            before_ids = set(id(p) for p in filtered)
            filtered2 = self._filter_peaks_dbscan(filtered)
            after_ids = set(id(p) for p in filtered2)
            removed = before_ids - after_ids
            if removed:
                for pk in filtered:
                    if id(pk) in removed:
                        _add_reject(
                            pk,
                            f"DBSCAN outlier(eps={float(self.dbscan_eps):.3g}, min_samples={int(self.dbscan_min_samples)})",
                        )
            filtered = filtered2

        if return_rejections:
            return filtered, list(rejected_map.values())
        return filtered

    def analyze_frame(
        self,
        frame: np.ndarray,
        prev_center: Optional[Tuple[float, float]] = None,
        criterion: str = "aic",
        center_override: Optional[Tuple[float, float]] = None,
        roi_size_override: Optional[int] = None,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
        initial_xy: Optional[Sequence[Tuple[float, float]]] = None,
        min_peaks: int = 2,
        max_peaks: int = 3,
    ) -> FrameAnalysis:
        use_roi_size = roi_size_override if roi_size_override is not None else self.roi_size
        frame_f = np.asarray(frame, dtype=np.float64)
        roi_mask = roi_mask_override

        initial_local: Optional[List[Tuple[float, float]]] = None
        seed_spots: Optional[List[Dict[str, float]]] = None
        roi_pre: np.ndarray
        roi_det: np.ndarray
        if roi_bounds_override is not None:
            roi_raw, origin = self._crop_rect(frame_f, roi_bounds_override)
            roi_pre = self._apply_roi_preprocess(roi_raw)
            roi_det = self._apply_detection(roi_pre)
            if initial_xy:
                initial_local = []
                h_roi, w_roi = roi_pre.shape
                for x_abs, y_abs in initial_xy:
                    try:
                        lx = float(x_abs) - float(origin[0])
                        ly = float(y_abs) - float(origin[1])
                    except Exception:
                        continue
                    if 0 <= lx < w_roi and 0 <= ly < h_roi:
                        initial_local.append((lx, ly))
                # Precheck behavior: snap manual seeds to strongest pixel nearby on roi_pre.
                if initial_local:
                    try:
                        r_pre = int(getattr(self, "precheck_radius_px", 0))
                    except Exception:
                        r_pre = 0
                    if r_pre > 0:
                        snapped = []
                        for lx, ly in initial_local:
                            iy = int(round(float(ly)))
                            ix = int(round(float(lx)))
                            iy2, ix2, _moved = self._snap_to_local_max_pixel(
                                roi_pre, iy, ix, radius=int(r_pre), roi_mask=roi_mask
                            )
                            snapped.append((float(ix2), float(iy2)))
                        initial_local = snapped
        else:
            if center_override is None:
                filtered_for_center = self._pipeline(frame_f)
                center = self._estimate_center(filtered_for_center, prev_center)
            else:
                center = center_override
            roi_raw, origin = self._crop_square(frame_f, center, use_roi_size)
            roi_pre = self._apply_roi_preprocess(roi_raw)
            roi_det = self._apply_detection(roi_pre)
            if initial_xy:
                initial_local = []
                h_roi, w_roi = roi_pre.shape
                for x_abs, y_abs in initial_xy:
                    try:
                        lx = float(x_abs) - float(origin[0])
                        ly = float(y_abs) - float(origin[1])
                    except Exception:
                        continue
                    if 0 <= lx < w_roi and 0 <= ly < h_roi:
                        initial_local.append((lx, ly))
                # Precheck behavior: snap manual seeds to strongest pixel nearby on roi_pre.
                if initial_local:
                    try:
                        r_pre = int(getattr(self, "precheck_radius_px", 0))
                    except Exception:
                        r_pre = 0
                    if r_pre > 0:
                        snapped = []
                        for lx, ly in initial_local:
                            iy = int(round(float(ly)))
                            ix = int(round(float(lx)))
                            iy2, ix2, _moved = self._snap_to_local_max_pixel(
                                roi_pre, iy, ix, radius=int(r_pre), roi_mask=roi_mask
                            )
                            snapped.append((float(ix2), float(iy2)))
                        initial_local = snapped

        # Compute detection seeds (independent of fitting sigma) for display.
        # - If manual initial_xy is provided, treat those as the seeds.
        # - Otherwise, for peak-based init_mode, compute seeds from the detection image.
        mode = (self.init_mode or "peak").strip().lower()
        if initial_local:
            seed_spots = [
                {"x": float(origin[0]) + float(lx), "y": float(origin[1]) + float(ly), "snr": 0.0}
                for (lx, ly) in initial_local
            ]
        elif mode.startswith("peak"):
            try:
                max_peaks_i = int(max_peaks)
            except Exception:
                max_peaks_i = 0
            max_peaks_i = max(1, int(max_peaks_i))
            coords_seed = self._top_pixels_by_value_nms(
                roi_det,
                max_peaks_i,
                roi_mask=roi_mask,
                min_distance=int(getattr(self, "peak_min_distance", 2)),
            )
            try:
                initial_local = [(float(x), float(y)) for (y, x) in coords_seed.tolist()]
            except Exception:
                initial_local = [(float(x), float(y)) for (y, x) in coords_seed]
            seed_spots = [
                {"x": float(origin[0]) + float(lx), "y": float(origin[1]) + float(ly), "snr": 0.0}
                for (lx, ly) in (initial_local or [])
            ]

        # Fit/evaluation is performed on the preprocessed ROI (median/open only),
        # while DoG/LoG is used only for peak initialization/snap.
        noise_sigma = self._estimate_noise_sigma(roi_pre, roi_mask)
        det_noise_sigma = self._estimate_noise_sigma(roi_det, roi_mask)
        models = {}
        min_peaks = max(1, int(min_peaks))
        max_peaks = max(min_peaks, int(max_peaks))
        for n_peaks in range(min_peaks, max_peaks + 1):
            models[n_peaks] = self._fit_model(
                roi_pre,
                origin,
                n_peaks,
                noise_sigma,
                roi_mask,
                initial_xy_local=initial_local if mode.startswith("peak") else initial_local,
                init_image=roi_det,
                init_noise_sigma=det_noise_sigma,
            )

        use_bic = criterion.lower() == "bic"
        best = None
        best_score = None
        for n_peaks, model in models.items():
            score = model.bic if use_bic else model.aic
            if best_score is None or score < best_score:
                best_score = score
                best = n_peaks

        # --- Post-processing filter for the best model ---
        if best is not None:
            best_model = models[best]
            # Keep identity map from pre-filter peaks -> original peak index.
            # This avoids any proximity-based matching in the UI.
            try:
                prefilter_peaks = list(best_model.peaks)
                for i, pk in enumerate(prefilter_peaks):
                    try:
                        setattr(pk, "_prefilter_index", int(i))
                    except Exception:
                        pass
                best_model.prefilter_index_by_id = {id(pk): i for i, pk in enumerate(prefilter_peaks)}
                best_model.prefilter_peaks = list(prefilter_peaks)
            except Exception:
                prefilter_peaks = list(best_model.peaks)
            filtered_peaks, rejected_infos = self._filter_peaks(
                best_model.peaks,
                origin,
                roi_pre.shape,
                roi_pre=roi_pre,
                roi_mask=roi_mask,
                return_rejections=True,
            )
            # Enforce minimum peak distance before snapping so snap itself does not
            # reduce the visible peak count by making peaks newly collide.
            if filtered_peaks:
                filtered_peaks, dist_rejected = self._enforce_min_peak_distance(
                    filtered_peaks, return_rejections=True
                )
                try:
                    rejected_infos.extend(list(dist_rejected))
                except Exception:
                    pass
            # Optionally snap best peaks to local maxima on the detection image (ROI-local).
            if self.fit_enabled and self.snap_enabled and filtered_peaks:
                filtered_peaks = self._snap_peaks_to_local_maxima(filtered_peaks, roi_det, origin, roi_mask=roi_mask)

            # Optional: re-fit once using snapped peak seeds, then filter/snap again.
            # This is OFF by default because it can change AIC/BIC and runtime.
            if self.fit_enabled and self.snap_refit_enabled and best is not None and filtered_peaks:
                try:
                    seeds_local = [(float(pk.x) - float(origin[0]), float(pk.y) - float(origin[1])) for pk in filtered_peaks]
                except Exception:
                    seeds_local = None
                if seeds_local:
                    refit = self._fit_model(
                        roi_pre,
                        origin,
                        best,
                        noise_sigma,
                        roi_mask,
                        initial_xy_local=seeds_local,
                        init_image=roi_det,
                        init_noise_sigma=det_noise_sigma,
                    )
                    models[best] = refit
                    best_model = refit
                    # Rebuild identity map for refit peaks (pre-filter).
                    try:
                        prefilter_peaks = list(best_model.peaks)
                        for i, pk in enumerate(prefilter_peaks):
                            try:
                                setattr(pk, "_prefilter_index", int(i))
                            except Exception:
                                pass
                        best_model.prefilter_index_by_id = {id(pk): i for i, pk in enumerate(prefilter_peaks)}
                        best_model.prefilter_peaks = list(prefilter_peaks)
                    except Exception:
                        pass
                    filtered_peaks, rejected_infos = self._filter_peaks(
                        best_model.peaks,
                        origin,
                        roi_pre.shape,
                        roi_pre=roi_pre,
                        roi_mask=roi_mask,
                        return_rejections=True,
                    )
                    if filtered_peaks:
                        filtered_peaks, dist_rejected = self._enforce_min_peak_distance(
                            filtered_peaks, return_rejections=True
                        )
                        try:
                            rejected_infos.extend(list(dist_rejected))
                        except Exception:
                            pass
                    if self.snap_enabled and filtered_peaks:
                        filtered_peaks = self._snap_peaks_to_local_maxima(filtered_peaks, roi_det, origin, roi_mask=roi_mask)

            filtered_peaks, rejected_infos = self._ensure_min_peak_count(
                filtered_peaks,
                prefilter_peaks,
                min_keep=min(min_peaks, best),
                rejected_infos=rejected_infos,
            )

            # Update the peaks in the best model result.
            # Note: We do not update best_n_peaks selection itself here.
            best_model.peaks = filtered_peaks
            try:
                best_model.excluded_infos = rejected_infos
            except Exception:
                pass
            self._force_min_peak_count_on_model(best_model, min_keep=min(min_peaks, best))

        return FrameAnalysis(
            best_n_peaks=best,
            criterion="bic" if use_bic else "aic",
            noise_sigma=noise_sigma,
            snr_threshold=self.snr_threshold,
            models=models,
            roi=roi_pre,
            origin=origin,
            roi_mask=roi_mask,
            seed_spots=seed_spots,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Backward-compat: apply current pipeline to full frame.
        return self._pipeline(frame)

    def compute_detection_image_full(
        self,
        frame: np.ndarray,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Full-size detection image used for peak detection.
        If ROI bounds are given, returns an array with NaN outside ROI (and outside ellipse mask if provided).
        """
        frame_f = np.asarray(frame, dtype=np.float64)
        if roi_bounds_override is None:
            return self._pipeline(frame_f)

        roi_raw, origin = self._crop_rect(frame_f, roi_bounds_override)
        roi_det = self._pipeline(roi_raw)
        full = np.full_like(frame_f, np.nan, dtype=np.float64)
        x0, y0 = int(origin[0]), int(origin[1])
        h, w = roi_det.shape
        full[y0 : y0 + h, x0 : x0 + w] = roi_det
        if roi_mask_override is not None:
            mask = roi_mask_override.astype(bool)
            crop = full[y0 : y0 + h, x0 : x0 + w]
            try:
                crop[~mask] = np.nan
            except Exception:
                pass
        return full

    def compute_roi_visual_images(
        self,
        frame: np.ndarray,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return two images for ROI display:
        - Left: preprocessed ROI image (median/open only; detection filter not applied)
        - Right: ROI image for detection (preprocess -> LoG/DoG)
        Returns (None, None) when no ROI is available.
        """
        if roi_bounds_override is None:
            return None, None
        frame_f = np.asarray(frame, dtype=np.float64)
        roi_raw, _origin = self._crop_rect(frame_f, roi_bounds_override)
        roi_pre = self._apply_roi_preprocess(roi_raw)
        roi_det = self._apply_detection(roi_pre)
        roi_pre_v = np.asarray(roi_pre, dtype=np.float64).copy()
        roi_det_v = np.asarray(roi_det, dtype=np.float64).copy()
        if roi_mask_override is not None:
            try:
                mask = roi_mask_override.astype(bool)
                roi_pre_v[~mask] = np.nan
                roi_det_v[~mask] = np.nan
            except Exception:
                pass
        return roi_pre_v, roi_det_v

    def _estimate_center(
        self,
        frame: np.ndarray,
        prev_center: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        if prev_center is None:
            cy, cx = ndimage.center_of_mass(np.clip(frame - np.min(frame), 0.0, None))
            if np.isnan(cx) or np.isnan(cy):
                h, w = frame.shape
                return w / 2.0, h / 2.0
            return float(cx), float(cy)

        small_roi, origin = self._crop_square(frame, prev_center, max(8, self.roi_size // 2))
        cy, cx = ndimage.center_of_mass(np.clip(small_roi - np.min(small_roi), 0.0, None))
        if np.isnan(cx) or np.isnan(cy):
            return prev_center
        return float(origin[0] + cx), float(origin[1] + cy)

    def _crop_square(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        size: int,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape
        half = size // 2
        cx = int(round(center[0]))
        cy = int(round(center[1]))
        x0 = max(cx - half, 0)
        x1 = min(cx + half + 1, w)
        y0 = max(cy - half, 0)
        y1 = min(cy + half + 1, h)
        return frame[y0:y1, x0:x1], (x0, y0)

    def _crop_rect(
        self,
        frame: np.ndarray,
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        x0, y0, w, h = bounds
        h_img, w_img = frame.shape
        x0_i = max(int(round(x0)), 0)
        y0_i = max(int(round(y0)), 0)
        x1_i = min(int(round(x0 + w)), w_img)
        y1_i = min(int(round(y0 + h)), h_img)
        return frame[y0_i:y1_i, x0_i:x1_i], (x0_i, y0_i)

    def _estimate_noise_sigma(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        data = frame[mask] if mask is not None else frame.ravel()
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        sigma = 1.4826 * mad  # robust estimator
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.std(data))
        return max(sigma, 1e-6)

    def _validate_peak_coords(
        self,
        coords: np.ndarray,
        intensity_image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        snap_to_local_max: bool = True,
        search_radius: int = 5,
    ) -> np.ndarray:
        """
        Snap/validate detected peak coordinates against an intensity image.

        This is crucial for DoG-based detection, which creates ring patterns and may
        detect valleys instead of true peaks.
        """
        if coords.shape[0] == 0:
            return coords

        h, w = intensity_image.shape
        out = []

        for y, x in coords:
            iy = int(y)
            ix = int(x)

            if not (0 <= iy < h and 0 <= ix < w):
                iy = int(np.clip(iy, 0, h - 1))
                ix = int(np.clip(ix, 0, w - 1))

            if snap_to_local_max:
                try:
                    r = int(search_radius)
                except Exception:
                    r = 0
                iy, ix, _moved = self._snap_to_local_max_pixel(
                    intensity_image, iy, ix, radius=r, roi_mask=roi_mask
                )

            out.append([int(iy), int(ix)])

        return np.asarray(out, dtype=np.int64)

    def _initial_params(
        self,
        frame: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        init_noise_sigma: Optional[float] = None,
        validate_image: Optional[np.ndarray] = None,
    ) -> List[float]:
        try:
            m0 = max(0, int(getattr(self, "margin", 0)))
        except Exception:
            m0 = 0
        try:
            h0, w0 = np.asarray(frame).shape[:2]
            base_border = max(3, int(min(int(h0), int(w0)) * 0.05))
        except Exception:
            base_border = 3

        def _iter_margin_masks(shape_hw: Tuple[int, int]):
            h, w = int(shape_hw[0]), int(shape_hw[1])
            base = max(3, int(min(h, w) * 0.05))
            for m in range(int(m0), -1, -1):
                use_border = int(max(base, m))
                yield use_border, self._roi_mask_with_margin(roi_mask, use_border, shape=(h, w))

        mode = (self.init_mode or "peak").strip().lower()
        if mode == "watershed":
            # When a validation image (typically preprocessed intensity) is provided,
            # prefer running watershed on that image rather than on DoG/LoG to avoid ring artifacts.
            ws_img = validate_image if validate_image is not None else frame
            params_ws = None
            used_mask = roi_mask
            try:
                ws_hw = np.asarray(ws_img).shape[:2]
            except Exception:
                ws_hw = (int(h0), int(w0))
            for use_border, mask_m in _iter_margin_masks((int(ws_hw[0]), int(ws_hw[1]))):
                params_ws = self._initial_params_watershed(
                    ws_img, n_peaks, roi_mask=mask_m, border_exclude=use_border
                )
                if params_ws is not None:
                    used_mask = mask_m
                    break
            if params_ws is not None:
                # If a validation image is provided (typically preprocessed intensity),
                # snap watershed seeds to true intensity maxima to avoid DoG/LoG ring artifacts.
                if validate_image is not None and len(params_ws) >= n_peaks * 4 + 1:
                    try:
                        entries = []
                        for i in range(int(n_peaks)):
                            amp, x, y, sigma0 = params_ws[i * 4 : (i + 1) * 4]
                            entries.append((float(amp), float(x), float(y), float(sigma0)))

                        # Build coords as (row=y, col=x)
                        coords = np.array([[int(round(e[2])), int(round(e[1]))] for e in entries], dtype=np.int64)
                        try:
                            r_pre = int(getattr(self, "precheck_radius_px", 0))
                        except Exception:
                            r_pre = 0
                        coords_v = self._validate_peak_coords(
                            coords,
                            np.asarray(validate_image, dtype=np.float64),
                            roi_mask=used_mask,
                            snap_to_local_max=True,
                            search_radius=max(1, int(r_pre)) if int(r_pre) > 0 else 5,
                        )

                        # Ensure we keep exactly n_peaks coords (pad if validation reduced them)
                        coords_final = []
                        seen = set()
                        for yy, xx in coords_v:
                            t = (int(yy), int(xx))
                            if t not in seen:
                                coords_final.append([int(yy), int(xx)])
                                seen.add(t)
                            if len(coords_final) >= int(n_peaks):
                                break

                        if len(coords_final) < int(n_peaks):
                            # Pad with remaining original coords ordered by validate_image intensity
                            imgv = np.asarray(validate_image, dtype=np.float64)
                            cand = []
                            for yy, xx in coords:
                                t = (int(yy), int(xx))
                                if t in seen:
                                    continue
                                if 0 <= int(yy) < imgv.shape[0] and 0 <= int(xx) < imgv.shape[1]:
                                    cand.append((float(imgv[int(yy), int(xx)]), int(yy), int(xx)))
                            cand.sort(key=lambda t: t[0], reverse=True)
                            for _v, yy, xx in cand:
                                coords_final.append([yy, xx])
                                seen.add((yy, xx))
                                if len(coords_final) >= int(n_peaks):
                                    break

                        # Rebuild params with snapped coords; keep sigma0, refresh amp from init frame.
                        img_init = np.asarray(frame, dtype=np.float64)
                        params: List[float] = []
                        for i in range(int(n_peaks)):
                            sigma0 = float(entries[i][3]) if i < len(entries) else float(self.initial_sigma)
                            yy, xx = coords_final[i] if i < len(coords_final) else coords[i].tolist()
                            yy_i = int(np.clip(int(yy), 0, img_init.shape[0] - 1))
                            xx_i = int(np.clip(int(xx), 0, img_init.shape[1] - 1))
                            amp0 = max(float(img_init[yy_i, xx_i]), 1e-6)
                            params.extend([amp0, float(xx_i), float(yy_i), sigma0])
                        params.append(float(np.nanmedian(img_init)))
                        return params
                    except Exception:
                        return params_ws
                return params_ws
            # fallback order
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob
        elif mode == "blob_doh":
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_doh = self._initial_params_blob_doh(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_doh is not None:
                    # If validate_image (typically roi_pre) is provided, snap DoH seeds to
                    # true intensity maxima to avoid LoG/DoG ring artifacts.
                    if validate_image is not None and len(params_doh) >= n_peaks * 4 + 1:
                        try:
                            entries = []
                            for i in range(int(n_peaks)):
                                amp, x, y, sigma0 = params_doh[i * 4 : (i + 1) * 4]
                                entries.append((float(amp), float(x), float(y), float(sigma0)))

                            coords = np.array([[int(round(e[2])), int(round(e[1]))] for e in entries], dtype=np.int64)
                            try:
                                r_pre = int(getattr(self, "precheck_radius_px", 0))
                            except Exception:
                                r_pre = 0
                            coords_v = self._validate_peak_coords(
                                coords,
                                np.asarray(validate_image, dtype=np.float64),
                                roi_mask=mask_m,
                                snap_to_local_max=True,
                                search_radius=max(1, int(r_pre)) if int(r_pre) > 0 else 5,
                            )

                            # Unique + cap to n_peaks
                            coords_final = []
                            seen = set()
                            for yy, xx in coords_v:
                                t = (int(yy), int(xx))
                                if t not in seen:
                                    coords_final.append([int(yy), int(xx)])
                                    seen.add(t)
                                if len(coords_final) >= int(n_peaks):
                                    break

                            # Optional NMS by peak_min_distance on snapped coords (Chebyshev window).
                            try:
                                r_nms = int(getattr(self, "peak_min_distance", 0) or 0)
                            except Exception:
                                r_nms = 0
                            r_nms = max(0, int(r_nms))
                            if r_nms > 0 and coords_final:
                                imgv = np.asarray(validate_image, dtype=np.float64)
                                scored = []
                                for yy, xx in coords_final:
                                    if 0 <= int(yy) < imgv.shape[0] and 0 <= int(xx) < imgv.shape[1]:
                                        scored.append((float(imgv[int(yy), int(xx)]), int(yy), int(xx)))
                                scored.sort(key=lambda t: t[0], reverse=True)
                                kept = []
                                for _v, yy, xx in scored:
                                    ok = True
                                    for ky, kx in kept:
                                        if abs(int(kx) - int(xx)) <= r_nms and abs(int(ky) - int(yy)) <= r_nms:
                                            ok = False
                                            break
                                    if ok:
                                        kept.append((int(yy), int(xx)))
                                    if len(kept) >= int(n_peaks):
                                        break
                                coords_final = [[yy, xx] for (yy, xx) in kept]

                            if len(coords_final) < int(n_peaks):
                                # Pad with original coords ordered by validate_image intensity
                                imgv = np.asarray(validate_image, dtype=np.float64)
                                cand = []
                                for yy, xx in coords:
                                    t = (int(yy), int(xx))
                                    if t in seen:
                                        continue
                                    if 0 <= int(yy) < imgv.shape[0] and 0 <= int(xx) < imgv.shape[1]:
                                        cand.append((float(imgv[int(yy), int(xx)]), int(yy), int(xx)))
                                cand.sort(key=lambda t: t[0], reverse=True)
                                for _v, yy, xx in cand:
                                    coords_final.append([yy, xx])
                                    seen.add((yy, xx))
                                    if len(coords_final) >= int(n_peaks):
                                        break

                            # Rebuild params with snapped coords; keep sigma0, refresh amp from validate_image.
                            imgv = np.asarray(validate_image, dtype=np.float64)
                            params: List[float] = []
                            for i in range(int(n_peaks)):
                                sigma0 = float(entries[i][3]) if i < len(entries) else float(self.initial_sigma)
                                yy, xx = coords_final[i] if i < len(coords_final) else coords[i].tolist()
                                yy_i = int(np.clip(int(yy), 0, imgv.shape[0] - 1))
                                xx_i = int(np.clip(int(xx), 0, imgv.shape[1] - 1))
                                amp0 = max(float(imgv[yy_i, xx_i]), 1e-6)
                                params.extend([amp0, float(xx_i), float(yy_i), sigma0])
                            params.append(float(np.nanmedian(imgv)))
                            return params
                        except Exception:
                            return params_doh
                    return params_doh
            # fallback to blob_log
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob
        elif mode == "multiscale":
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_multi = self._initial_params_multiscale(
                    frame, n_peaks, roi_mask=mask_m, border_exclude=use_border
                )
                if params_multi is not None:
                    return params_multi
            # fallback to blob_log
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob
        elif mode == "blob":
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob

        # Peak-based initialization: choose by descending pixel value with NMS (min_distance).
        # This makes the initializer deterministic and aligned with the displayed LoG/DoG image.
        if mode.startswith("peak"):
            use_subpixel = "subpixel" in mode
            coords = self._top_pixels_by_value_nms(
                frame,
                n_peaks,
                roi_mask=roi_mask,
                min_distance=int(getattr(self, "peak_min_distance", 2)),
            )

            params: List[float] = []
            for y, x in coords[:n_peaks]:
                iy = int(y)
                ix = int(x)
                amp = max(float(frame[iy, ix]), 1e-6)
                if use_subpixel:
                    rx, ry = self._refine_subpixel_quadratic(frame, ix, iy)
                    params.extend([amp, float(rx), float(ry), float(self.initial_sigma)])
                else:
                    params.extend([amp, float(ix), float(iy), float(self.initial_sigma)])
            while len(params) < n_peaks * 4:
                params.extend([1.0, frame.shape[1] / 2.0, frame.shape[0] / 2.0, float(self.initial_sigma)])
            params.append(float(np.nanmedian(frame)))
            return params

        use_subpixel = "subpixel" in mode

        coords, used_mask, _used_m = self._detect_local_maxima(
            frame,
            n_peaks,
            roi_mask=roi_mask,
            threshold_rel=self.localmax_threshold_rel,
            noise_sigma=init_noise_sigma,
            return_meta=True,
        )

        # Validate/snap peaks against the intensity image if provided
        # This is critical for DoG-based detection which creates ring patterns
        if validate_image is not None and coords.shape[0] > 0:
            try:
                r_pre = int(getattr(self, "precheck_radius_px", 0))
            except Exception:
                r_pre = 0
            coords = self._validate_peak_coords(
                coords,
                validate_image,
                used_mask,
                snap_to_local_max=True,
                search_radius=max(1, int(r_pre)) if int(r_pre) > 0 else 5,
            )

        if coords.shape[0] < n_peaks:
            # Fallback: pick strongest pixels globally, but prefer excluding ROI margin.
            img = np.asarray(frame, dtype=np.float64)
            coords_best = None
            for use_border, mask_m in _iter_margin_masks((int(img.shape[0]), int(img.shape[1]))):
                try:
                    if mask_m is not None and mask_m.shape == img.shape:
                        masked = np.where(mask_m.astype(bool), img, -np.inf)
                    else:
                        masked = img
                        if use_border > 0 and img.shape[0] > 2 * use_border and img.shape[1] > 2 * use_border:
                            masked = masked.copy()
                            masked[:use_border, :] = -np.inf
                            masked[-use_border:, :] = -np.inf
                            masked[:, :use_border] = -np.inf
                            masked[:, -use_border:] = -np.inf
                    flat_idx = np.argsort(masked.ravel())[::-1][: int(n_peaks)]
                    ys, xs = np.unravel_index(flat_idx, img.shape)
                    coords_try = np.stack([ys, xs], axis=1)
                except Exception:
                    continue
                # accept if we have enough finite pixels
                try:
                    ok = 0
                    for yy, xx in coords_try:
                        v = float(img[int(yy), int(xx)])
                        if np.isfinite(v):
                            ok += 1
                    if ok >= int(n_peaks):
                        coords_best = coords_try
                        break
                except Exception:
                    coords_best = coords_try
                    break
            if coords_best is not None:
                coords = np.asarray(coords_best, dtype=np.int64)

        params: List[float] = []
        for y, x in coords[:n_peaks]:
            iy = int(y)
            ix = int(x)
            amp = max(float(frame[iy, ix]), 1e-6)
            if use_subpixel:
                rx, ry = self._refine_subpixel_quadratic(frame, ix, iy)
                params.extend([amp, float(rx), float(ry), float(self.initial_sigma)])
            else:
                params.extend([amp, float(ix), float(iy), float(self.initial_sigma)])
        while len(params) < n_peaks * 4:
            params.extend([1.0, frame.shape[1] / 2.0, frame.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.median(frame)))
        return params

    def _initial_params_watershed(
        self,
        frame: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        border_exclude: int = 0,
    ) -> Optional[List[float]]:
        """
        Watershed-based initializer:
        - h-maxima suppresses shallow maxima (noise)
        - watershed segments basins around markers
        - regionprops yields centroid/size to seed (x,y,sigma)
        Returns None on failure so caller can fall back.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None
        h, w = img.shape
        v_max = float(np.nanmax(img))
        v_min = float(np.nanmin(img))
        if not np.isfinite(v_max) or not np.isfinite(v_min) or v_max <= v_min:
            return None

        h_rel = float(self.watershed_h_rel)
        h_rel = min(max(h_rel, 0.0), 1.0)

        # Adaptive h-maxima: use noise-based threshold if enabled
        if self.watershed_adaptive_h:
            try:
                noise_sigma = self._estimate_noise_sigma(img, roi_mask)
                h_val_adaptive = max(3.0 * noise_sigma, h_rel * (v_max - v_min))
                h_val = h_val_adaptive
            except Exception:
                h_val = h_rel * (v_max - v_min)
        else:
            h_val = h_rel * (v_max - v_min)

        if h_val <= 0:
            return None

        mask = roi_mask.astype(bool) if roi_mask is not None else None

        try:
            maxima = morphology.h_maxima(img, h=h_val)
        except Exception:
            return None
        if maxima is None:
            return None
        maxima = maxima.astype(bool)
        # Exclude border maxima to reduce LoG/DoG edge artifacts.
        try:
            border = max(1, int(self.peak_min_distance), int(border_exclude))
        except Exception:
            border = max(1, int(border_exclude))
        if border > 0 and h > 2 * border and w > 2 * border:
            maxima[:border, :] = False
            maxima[-border:, :] = False
            maxima[:, :border] = False
            maxima[:, -border:] = False
        if mask is not None:
            try:
                maxima &= mask
            except Exception:
                pass
        if not np.any(maxima):
            return None

        # pick strongest maxima as markers (to avoid over-segmentation)
        ys, xs = np.nonzero(maxima)
        if ys.size == 0:
            return None
        resp = img[ys, xs]
        order = np.argsort(resp)[::-1]
        # allow more markers than peaks, but cap
        max_markers = int(max(20, n_peaks * 5))
        order = order[:max_markers]
        markers = np.zeros_like(img, dtype=np.int32)
        m_id = 1
        for idx in order:
            y = int(ys[idx])
            x = int(xs[idx])
            # enforce min_distance roughly by skipping close markers
            # (cheap check against existing markers)
            if self.peak_min_distance > 1:
                r = int(self.peak_min_distance)
                y0 = max(0, y - r)
                y1 = min(h, y + r + 1)
                x0 = max(0, x - r)
                x1 = min(w, x + r + 1)
                if np.any(markers[y0:y1, x0:x1] > 0):
                    continue
            markers[y, x] = m_id
            m_id += 1
        if m_id <= 1:
            return None

        try:
            labels = segmentation.watershed(-img, markers, mask=mask)
        except Exception:
            return None
        if labels is None:
            return None

        # build candidate peaks from regions
        try:
            props = measure.regionprops(labels, intensity_image=img)
        except Exception:
            return None
        if not props:
            return None

        sigma_min, sigma_max = self.sigma_bounds
        candidates = []
        for rp in props:
            if rp.label == 0:
                continue
            # Representative point:
            # Use the brightest pixel inside the watershed region (on roi_pre/intensity image),
            # rather than centroid. Centroid can land on a non-peak area for skewed regions.
            try:
                img_box = np.asarray(getattr(rp, "intensity_image", None), dtype=np.float64)
            except Exception:
                img_box = None
            try:
                reg_mask = np.asarray(getattr(rp, "image", None), dtype=bool)
            except Exception:
                reg_mask = None
            cx = cy = float("nan")
            amp = float(getattr(rp, "max_intensity", 0.0))
            if img_box is not None and reg_mask is not None and img_box.shape == reg_mask.shape and img_box.size > 0:
                try:
                    masked = np.where(reg_mask, img_box, -np.inf)
                    flat_idx = int(np.nanargmax(masked))
                    iy_loc, ix_loc = divmod(flat_idx, int(masked.shape[1]))
                    # bbox: (min_row, min_col, max_row, max_col)
                    b = getattr(rp, "bbox", None)
                    if b is not None and len(b) >= 2:
                        cy = float(int(b[0]) + int(iy_loc))
                        cx = float(int(b[1]) + int(ix_loc))
                    else:
                        cy = float(iy_loc)
                        cx = float(ix_loc)
                    try:
                        amp = float(masked[int(iy_loc), int(ix_loc)])
                    except Exception:
                        amp = float(getattr(rp, "max_intensity", 0.0))
                except Exception:
                    # Fallback: keep centroid if something goes wrong
                    try:
                        cy, cx = rp.centroid  # (row, col)
                    except Exception:
                        pass
            else:
                # Fallback: centroid
                try:
                    cy, cx = rp.centroid  # (row, col)
                except Exception:
                    pass
            eq_d = float(getattr(rp, "equivalent_diameter", 0.0))
            if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(amp):
                continue
            if eq_d <= 0 or not np.isfinite(eq_d):
                sigma0 = float(self.initial_sigma)
            else:
                sigma0 = eq_d / 4.0
            sigma0 = min(max(float(sigma0), float(sigma_min)), float(sigma_max))
            candidates.append((amp, float(cx), float(cy), sigma0))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)

        # Enforce peak spacing on the chosen representative points as well.
        # Even after watershed, nearby regions can produce very close maxima; apply a final greedy NMS
        # using the same window notion as Peak initializer (Chebyshev distance <= r).
        try:
            r_nms = int(getattr(self, "peak_min_distance", 0) or 0)
        except Exception:
            r_nms = 0
        r_nms = max(0, int(r_nms))
        chosen = []
        if r_nms > 0:
            for amp, x, y, sigma0 in candidates:
                ok = True
                for _amp2, x2, y2, _s2 in chosen:
                    if abs(float(x2) - float(x)) <= float(r_nms) and abs(float(y2) - float(y)) <= float(r_nms):
                        ok = False
                        break
                if ok:
                    chosen.append((amp, x, y, sigma0))
                if len(chosen) >= int(n_peaks):
                    break
        else:
            chosen = list(candidates[: int(n_peaks)])

        params: List[float] = []
        for amp, x, y, sigma0 in chosen:
            params.extend([max(float(amp), 1e-6), x, y, sigma0])
        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])
        params.append(float(np.nanmedian(img)))
        return params

    def _initial_params_blob(
        self, frame: np.ndarray, n_peaks: int, roi_mask: Optional[np.ndarray] = None, border_exclude: int = 0
    ) -> Optional[List[float]]:
        """
        Multiscale blob detection based initializer using `skimage.feature.blob_log`.
        Returns None if blob candidates are not usable and the caller should fall back.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None

        v_max = float(np.nanmax(img))
        if not np.isfinite(v_max) or v_max <= 0:
            return None

        thr_rel = float(self.blob_threshold_rel)
        thr_rel = min(max(thr_rel, 0.0), 1.0)
        threshold = thr_rel * v_max

        min_sigma = max(float(self.blob_min_sigma), 0.1)
        max_sigma = max(float(self.blob_max_sigma), min_sigma + 1e-6)
        num_sigma = max(1, int(self.blob_num_sigma))
        overlap = float(self.blob_overlap)
        overlap = min(max(overlap, 0.0), 0.99)

        try:
            blobs = feature.blob_log(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
                overlap=overlap,
            )
        except Exception:
            return None

        if blobs is None or len(blobs) == 0:
            return None

        h, w = img.shape
        mask = roi_mask.astype(bool) if roi_mask is not None else None
        try:
            b = int(border_exclude)
        except Exception:
            b = 0
        b = max(0, b)
        candidates = []
        for y, x, s in blobs:
            ix = int(round(float(x)))
            iy = int(round(float(y)))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if b > 0 and (ix < b or iy < b or ix >= w - b or iy >= h - b):
                continue
            if mask is not None:
                try:
                    if not bool(mask[iy, ix]):
                        continue
                except Exception:
                    pass
            resp = float(img[iy, ix])
            candidates.append((resp, float(x), float(y), float(s)))

        if not candidates:
            return None

        # Sort by response descending
        candidates.sort(key=lambda t: t[0], reverse=True)

        sigma_min, sigma_max = self.sigma_bounds
        params: List[float] = []
        for resp, x, y, s in candidates[: int(n_peaks)]:
            amp = max(float(resp), 1e-6)
            sigma0 = float(s)
            if not np.isfinite(sigma0):
                sigma0 = float(self.initial_sigma)
            sigma0 = min(max(sigma0, float(sigma_min)), float(sigma_max))
            params.extend([amp, float(x), float(y), sigma0])

        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.nanmedian(img)))
        return params

    def _initial_params_blob_doh(
        self, frame: np.ndarray, n_peaks: int, roi_mask: Optional[np.ndarray] = None, border_exclude: int = 0
    ) -> Optional[List[float]]:
        """
        Determinant of Hessian (DoH) based blob detection initializer.
        DoH is computationally more efficient than LoG and robust for certain structures.
        Returns None if blob candidates are not usable and the caller should fall back.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None

        v_max = float(np.nanmax(img))
        if not np.isfinite(v_max) or v_max <= 0:
            return None

        thr_rel = float(self.blob_doh_threshold_rel)
        thr_rel = min(max(thr_rel, 0.0), 1.0)
        threshold = thr_rel * v_max

        min_sigma = max(float(self.blob_doh_min_sigma), 0.1)
        max_sigma = max(float(self.blob_doh_max_sigma), min_sigma + 1e-6)
        num_sigma = max(1, int(self.blob_doh_num_sigma))

        try:
            blobs = feature.blob_doh(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
            )
        except Exception:
            return None

        if blobs is None or len(blobs) == 0:
            return None

        h, w = img.shape
        mask = roi_mask.astype(bool) if roi_mask is not None else None
        try:
            b = int(border_exclude)
        except Exception:
            b = 0
        b = max(0, b)
        candidates = []
        for y, x, s in blobs:
            ix = int(round(float(x)))
            iy = int(round(float(y)))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if b > 0 and (ix < b or iy < b or ix >= w - b or iy >= h - b):
                continue
            if mask is not None:
                try:
                    if not bool(mask[iy, ix]):
                        continue
                except Exception:
                    pass
            resp = float(img[iy, ix])
            candidates.append((resp, float(x), float(y), float(s)))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)

        sigma_min, sigma_max = self.sigma_bounds
        params: List[float] = []
        for resp, x, y, s in candidates[: int(n_peaks)]:
            amp = max(float(resp), 1e-6)
            sigma0 = float(s)
            if not np.isfinite(sigma0):
                sigma0 = float(self.initial_sigma)
            sigma0 = min(max(sigma0, float(sigma_min)), float(sigma_max))
            params.extend([amp, float(x), float(y), sigma0])

        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.nanmedian(img)))
        return params

    def _initial_params_multiscale(
        self, frame: np.ndarray, n_peaks: int, roi_mask: Optional[np.ndarray] = None, border_exclude: int = 0
    ) -> Optional[List[float]]:
        """
        Multiscale LoG-based peak detection that combines results from multiple sigma values.
        This helps detect spots of varying sizes robustly.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None

        sigmas = self.multiscale_sigmas
        if not sigmas:
            sigmas = [1.0, 1.6, 2.5, 4.0]

        all_peaks = []
        for sigma in sigmas:
            try:
                log_img = -ndimage.gaussian_laplace(img, sigma=float(sigma))
                coords = feature.peak_local_max(
                    log_img,
                    min_distance=int(self.peak_min_distance),
                    num_peaks=n_peaks * 3,
                    exclude_border=False,
                    threshold_rel=self.localmax_threshold_rel,
                )
                if coords is not None and len(coords) > 0:
                    for y, x in coords:
                        iy, ix = int(y), int(x)
                        if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                            resp = float(img[iy, ix])
                            all_peaks.append((resp, float(ix), float(iy), float(sigma)))
            except Exception:
                continue

        if not all_peaks:
            return None

        # Apply DBSCAN to remove duplicates and spatial outliers
        if len(all_peaks) > 1:
            coords_array = np.array([[p[1], p[2]] for p in all_peaks])
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=self.peak_min_distance * 1.5, min_samples=1).fit(coords_array)
                labels = clustering.labels_

                # For each cluster, keep the peak with highest response
                unique_labels = set(labels)
                clustered_peaks = []
                for label in unique_labels:
                    if label == -1:
                        continue
                    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
                    best_idx = max(cluster_indices, key=lambda i: all_peaks[i][0])
                    clustered_peaks.append(all_peaks[best_idx])
                all_peaks = clustered_peaks
            except Exception:
                pass

        # Sort by response descending
        all_peaks.sort(key=lambda t: t[0], reverse=True)

        # Filter by ROI mask
        h, w = img.shape
        mask = roi_mask.astype(bool) if roi_mask is not None else None
        try:
            b = int(border_exclude)
        except Exception:
            b = 0
        b = max(0, b)
        filtered_peaks = []
        for resp, x, y, s in all_peaks:
            ix = int(round(float(x)))
            iy = int(round(float(y)))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if b > 0 and (ix < b or iy < b or ix >= w - b or iy >= h - b):
                continue
            if mask is not None:
                try:
                    if not bool(mask[iy, ix]):
                        continue
                except Exception:
                    pass
            filtered_peaks.append((resp, x, y, s))

        if not filtered_peaks:
            return None

        sigma_min, sigma_max = self.sigma_bounds
        params: List[float] = []
        for resp, x, y, s in filtered_peaks[: int(n_peaks)]:
            amp = max(float(resp), 1e-6)
            sigma0 = float(s)
            if not np.isfinite(sigma0):
                sigma0 = float(self.initial_sigma)
            sigma0 = min(max(sigma0, float(sigma_min)), float(sigma_max))
            params.extend([amp, float(x), float(y), sigma0])

        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.nanmedian(img)))
        return params

    def _filter_peaks_dbscan(
        self, peaks: List[PeakStat], eps: Optional[float] = None, min_samples: Optional[int] = None
    ) -> List[PeakStat]:
        """
        Apply DBSCAN clustering to filter out spatial outliers (noise-derived peaks).
        Isolated peaks that don't form clusters are considered outliers and removed.
        """
        if not peaks or len(peaks) <= 1:
            return peaks

        eps_val = eps if eps is not None else self.dbscan_eps
        min_samp = min_samples if min_samples is not None else self.dbscan_min_samples

        coords = np.array([[pk.x, pk.y] for pk in peaks])

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=float(eps_val), min_samples=int(min_samp)).fit(coords)
            labels = clustering.labels_

            # Keep only peaks that are not labeled as noise (-1)
            filtered = [pk for pk, label in zip(peaks, labels) if label != -1]
            return filtered if filtered else peaks
        except Exception:
            return peaks

    def _build_bounds(self, frame: np.ndarray, n_peaks: int) -> Tuple[List[float], List[float]]:
        h, w = frame.shape
        lower: List[float] = []
        upper: List[float] = []
        for _ in range(n_peaks):
            lower.extend([0.0, 0.0, 0.0, self.sigma_bounds[0]])
            upper.extend([np.inf, float(w), float(h), self.sigma_bounds[1]])
        lower.append(-np.inf)  # offset
        upper.append(np.inf)
        return lower, upper

    def _build_bounds_seeded(
        self,
        frame: np.ndarray,
        n_peaks: int,
        seeds_xy_local: Optional[Sequence[Tuple[float, float]]] = None,
        max_shift_px: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Bounds builder that optionally constrains (x,y) around per-peak seeds.
        If seeds are not provided or max_shift_px <= 0, falls back to wide bounds.
        """
        h, w = frame.shape
        r = int(self.refit_max_shift_px if max_shift_px is None else max_shift_px)
        if seeds_xy_local is None or r <= 0:
            return self._build_bounds(frame, n_peaks)

        seeds = list(seeds_xy_local)
        lower: List[float] = []
        upper: List[float] = []
        for i in range(int(n_peaks)):
            # amplitude >= 0
            lower.append(0.0)
            upper.append(np.inf)

            if i < len(seeds):
                sx, sy = seeds[i]
                try:
                    sx = float(sx)
                    sy = float(sy)
                except Exception:
                    sx = float(w) / 2.0
                    sy = float(h) / 2.0
                x_lo = max(0.0, sx - float(r))
                x_hi = min(float(w), sx + float(r))
                y_lo = max(0.0, sy - float(r))
                y_hi = min(float(h), sy + float(r))
                # guard: ensure strict ordering for curve_fit bounds
                if x_hi <= x_lo:
                    x_lo, x_hi = 0.0, float(w)
                if y_hi <= y_lo:
                    y_lo, y_hi = 0.0, float(h)
            else:
                x_lo, x_hi = 0.0, float(w)
                y_lo, y_hi = 0.0, float(h)

            lower.extend([x_lo, y_lo, self.sigma_bounds[0]])
            upper.extend([x_hi, y_hi, self.sigma_bounds[1]])

        lower.append(-np.inf)  # offset
        upper.append(np.inf)
        return lower, upper

    def _multi_gaussian(self, xy_mesh: Tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        x, y = xy_mesh
        offset = params[-1]
        g = np.full_like(x, offset, dtype=np.float64)
        core = params[:-1]
        for i in range(0, len(core), 4):
            amp, x0, y0, sigma = core[i : i + 4]
            denom = 2.0 * (sigma**2 + 1e-12)
            g += amp * np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / denom))
        return g.ravel()

    def _fit_model(
        self,
        roi: np.ndarray,
        origin: Tuple[int, int],
        n_peaks: int,
        noise_sigma: float,
        roi_mask: Optional[np.ndarray] = None,
        initial_xy_local: Optional[List[Tuple[float, float]]] = None,
        init_image: Optional[np.ndarray] = None,
        init_noise_sigma: Optional[float] = None,
    ) -> ModelSelectionResult:
        h, w = roi.shape
        x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        p0: List[float]
        if initial_xy_local:
            # Prefer provided seed points (x,y in ROI-local coords)
            seeds: List[Tuple[float, float]] = []
            for lx, ly in initial_xy_local[: int(n_peaks)]:
                try:
                    fx = float(lx)
                    fy = float(ly)
                except Exception:
                    continue
                # keep within bounds
                fx = float(np.clip(fx, 0.0, float(w - 1)))
                fy = float(np.clip(fy, 0.0, float(h - 1)))
                seeds.append((fx, fy))

            sigma0 = float(self.initial_sigma)
            p0 = []
            for fx, fy in seeds:
                ix = int(round(fx))
                iy = int(round(fy))
                try:
                    amp0 = float(roi[iy, ix])
                except Exception:
                    amp0 = 1.0
                p0.extend([max(float(amp0), 1e-6), float(fx), float(fy), sigma0])
            while len(p0) < n_peaks * 4:
                p0.extend([1.0, roi.shape[1] / 2.0, roi.shape[0] / 2.0, float(self.initial_sigma)])
            p0.append(float(np.nanmedian(roi)))
        else:
            init_img = init_image if init_image is not None else roi
            # If using a detection image (DoG/LoG), validate peaks against the fit target image
            validate_img = roi if init_image is not None else None
            p0 = self._initial_params(
                init_img,
                n_peaks,
                roi_mask=roi_mask,
                init_noise_sigma=init_noise_sigma,
                validate_image=validate_img,
            )
            # If initialization was performed on detection image (DoG/LoG),
            # adjust amplitude/offset guesses using the fit target image.
            if init_image is not None and len(p0) >= n_peaks * 4 + 1:
                try:
                    for i in range(int(n_peaks)):
                        x0 = float(p0[i * 4 + 1])
                        y0 = float(p0[i * 4 + 2])
                        ix = int(round(x0))
                        iy = int(round(y0))
                        if 0 <= ix < w and 0 <= iy < h:
                            p0[i * 4 + 0] = max(float(roi[iy, ix]), 1e-6)
                    p0[-1] = float(np.nanmedian(roi))
                except Exception:
                    pass
        # Prevent peak collapse by constraining (x,y) around the initializer seeds.
        # If manual seeds are provided, they take precedence.
        seeds_for_bounds: Optional[List[Tuple[float, float]]] = None
        if initial_xy_local:
            seeds_for_bounds = list(initial_xy_local)
        elif len(p0) >= n_peaks * 4:
            try:
                seeds_for_bounds = [(float(p0[i * 4 + 1]), float(p0[i * 4 + 2])) for i in range(int(n_peaks))]
            except Exception:
                seeds_for_bounds = None
        bounds = self._build_bounds_seeded(
            roi,
            n_peaks,
            seeds_xy_local=seeds_for_bounds,
            max_shift_px=self.refit_max_shift_px,
        )

        init_peaks: List[PeakStat] = []
        if len(p0) >= n_peaks * 4:
            for i in range(n_peaks):
                amp0, x0, y0, sigma0 = p0[i * 4 : (i + 1) * 4]
                init_peaks.append(
                    PeakStat(
                        amplitude=float(amp0),
                        x=float(origin[0] + x0),
                        y=float(origin[1] + y0),
                        sigma=float(sigma0),
                        snr=float(amp0) / max(noise_sigma, 1e-6),
                    )
                )

        fit_applied = bool(getattr(self, "fit_enabled", True))
        if fit_applied:
            try:
                if roi_mask is not None:
                    mask = roi_mask.astype(bool)
                    x_fit = x[mask].ravel()
                    y_fit = y[mask].ravel()
                    z_fit = roi[mask].ravel()
                else:
                    x_fit = x.ravel()
                    y_fit = y.ravel()
                    z_fit = roi.ravel()
                popt, _ = curve_fit(
                    self._multi_gaussian,
                    (x_fit, y_fit),
                    z_fit,
                    p0=p0,
                    bounds=bounds,
                    maxfev=self.max_iterations,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Gaussian fit failed (n=%s): %s", n_peaks, exc)
                popt = np.asarray(p0, dtype=np.float64)
        else:
            popt = np.asarray(p0, dtype=np.float64)

        model = self._multi_gaussian((x, y), *popt).reshape(roi.shape)
        residuals = roi - model
        if roi_mask is not None:
            mask = roi_mask.astype(bool)
            residuals_eval = residuals[mask]
        else:
            residuals_eval = residuals.ravel()
        rss = float(np.sum(residuals_eval**2))
        residual_std = float(np.std(residuals_eval))

        sigma = max(noise_sigma, 1e-6)
        loglike = -0.5 * np.sum((residuals_eval / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))

        k = len(popt)
        n = int(np.sum(roi_mask)) if roi_mask is not None else roi.size
        aic = -2.0 * loglike + 2.0 * k
        bic = -2.0 * loglike + k * np.log(n)

        peaks: List[PeakStat] = []
        for i in range(n_peaks):
            amp, x0, y0, sigma_i = popt[i * 4 : (i + 1) * 4]
            peaks.append(
                PeakStat(
                    amplitude=float(amp),
                    x=float(origin[0] + x0),
                    y=float(origin[1] + y0),
                    sigma=float(sigma_i),
                    snr=float(amp) / max(noise_sigma, 1e-6),
                )
            )

        return ModelSelectionResult(
            n_peaks=n_peaks,
            popt=popt,
            rss=rss,
            loglike=float(loglike),
            aic=float(aic),
            bic=float(bic),
            residual_std=residual_std,
            fit_applied=fit_applied,
            init_peaks=init_peaks,
            peaks=peaks,
        )


__all__ = [
    "SpotAnalysis",
    "FrameAnalysis",
    "ModelSelectionResult",
    "PeakStat",
    "SpotAnalysisWindow",
    "PLUGIN_NAME",
    "create_plugin",
]


class SpotVisualizationWindow(QtWidgets.QMainWindow):
    """
    Window for visualizing analysis results, including ROI images and detected peaks.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spot Analysis Visualization")
        self.resize(600, 500)
        
        # Main widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Matplotlib Figure
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.cbar = None
        
        # Standard Matplotlib toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

    def update_view(self, roi: np.ndarray, result: FrameAnalysis, origin: Tuple[int, int]):
        """Draw the ROI image and fitting results"""
        self.ax.clear()
        self._safe_remove_colorbar()
        self.figure.subplots_adjust(right=0.80)
        # Background image
        im = self.ax.imshow(roi, cmap='viridis', origin='lower')
        self.cbar = self.figure.colorbar(im, ax=self.ax, label='Height (nm)')
        
        # Plot peaks from the selected model
        best_model = result.models[result.best_n_peaks]
        colors = ["magenta", "white", "orange", "cyan", "yellow", "lime", "red", "deepskyblue"]
        
        for i, pk in enumerate(best_model.peaks):
            # PeakStat coordinates are global coordinates, so convert them back to ROI-local coordinates.
            rx = pk.x - origin[0]
            ry = pk.y - origin[1]
            
            label = f"P{i+1}: S/N={pk.snr:.1f}"
            
            self.ax.plot(rx, ry, 'x', color=colors[i % len(colors)], markersize=10, markeredgewidth=2, label=label)

        # It may be useful to also plot other models with dotted markers
        # when comparing both 2-peak and 3-peak models,
        # but for now only the selected model is displayed.
        
        self.ax.set_title(f"Model: {result.best_n_peaks} peaks ({result.criterion.upper()})")
        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )
        self.ax.set_xlabel("X (pixel)")
        self.ax.set_ylabel("Y (pixel)")
        # Do not show the Z scale bar
        
        self.canvas.draw()

    def _safe_remove_colorbar(self):
        """Safely remove the colorbar to avoid Matplotlib KeyError issues"""
        try:
            if self.cbar:
                self.cbar.remove()
        except Exception:
            pass
        self.cbar = None


class SpotFullImageWindow(QtWidgets.QMainWindow):
    """
    Window for overlaying peak positions on the full AFM image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spot Analysis: Full Overlay Image")
        self.resize(700, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        gs = self.figure.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.25)
        self.ax_afm = self.figure.add_subplot(gs[0, :])
        self.ax_roi_pre = self.figure.add_subplot(gs[1, 0])
        self.ax_roi_det = self.figure.add_subplot(gs[1, 1])
        self.cbar = None
        self.selector = None
        self.on_select_callback = None
        self.edit_handler = None
        self._dragging = False
        self._drag_index = None
        self._roi_selector_paused = False

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def set_edit_handler(self, handler):
        """Forward edit events to SpotAnalysisWindow"""
        self.edit_handler = handler

    def enable_roi_selector(self, shape: str, callback):
        """Enable ROI selection and notify callback(dict) with the selection result"""
        from matplotlib.widgets import RectangleSelector, EllipseSelector

        self.on_select_callback = callback

        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None

        def _emit_rect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            w = xmax - xmin
            h = ymax - ymin
            if callback:
                callback({
                    "shape": "Rectangle",
                    "x0": xmin,
                    "y0": ymin,
                    "w": w,
                    "h": h,
                })

        def _emit_ellipse(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            rx = (xmax - xmin) / 2.0
            ry = (ymax - ymin) / 2.0
            if callback:
                callback({
                    "shape": "Ellipse",
                    "cx": cx,
                    "cy": cy,
                    "rx": rx,
                    "ry": ry,
                    "x0": xmin,
                    "y0": ymin,
                    "w": xmax - xmin,
                    "h": ymax - ymin,
                })

        if shape == "Ellipse (Circle)":
            self.selector = EllipseSelector(
                self.ax_afm,
                _emit_ellipse,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )
        else:
            self.selector = RectangleSelector(
                self.ax_afm,
                _emit_rect,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )

    def update_view(
        self,
        frame: np.ndarray,
        result: Optional[FrameAnalysis],
        spots: Optional[List[Dict[str, float]]] = None,
        initial_spots: Optional[List[Dict[str, float]]] = None,
        roi_overlay: Optional[Dict[str, float]] = None,
        spot_radius_px: Optional[float] = None,
        roi_pre_image: Optional[np.ndarray] = None,
        roi_det_image: Optional[np.ndarray] = None,
        det_label: Optional[str] = None,
        show_roi_spots: bool = True,
        show_det_spots: bool = True,
        show_initial_spots: bool = False,
        show_fit_spots_on_roi_pre: bool = False,
        init_mode: Optional[str] = None,
    ) -> None:
        """Draw top: full AFM image / bottom: preprocessed ROI (left) + detection ROI (right)"""
        # NOTE: Avoid UnboundLocalError if 'np' becomes a local binding unexpectedly.
        import numpy as np

        self.figure.subplots_adjust(right=0.82)
        self.ax_afm.clear()
        self.ax_roi_pre.clear()
        self.ax_roi_det.clear()
        self._safe_remove_colorbar()

        # --- AFM (top) ---
        self.ax_afm.imshow(frame, cmap="viridis", origin="lower")

        # --- ROI images (bottom) ---
        pre = roi_pre_image
        det = roi_det_image
        if pre is None:
            pre = np.full((2, 2), np.nan, dtype=np.float64)
        if det is None:
            det = np.full((2, 2), np.nan, dtype=np.float64)

        pre_cmap = plt.get_cmap("viridis").copy()
        det_cmap = plt.get_cmap("magma").copy()
        for cm in (pre_cmap, det_cmap):
            try:
                cm.set_bad(color=(0, 0, 0, 0))
            except Exception:
                pass
        self.ax_roi_pre.imshow(pre, cmap=pre_cmap, origin="lower")
        self.ax_roi_det.imshow(det, cmap=det_cmap, origin="lower")

        # Determine detection mode for conditional overlays.
        # - Preprocessed: det_label starts with "Pre"
        # - DoG/LoG: otherwise (when det_label is missing, assume DoG/LoG-like behavior)
        _dl = (det_label or "").strip().lower()
        is_pre = _dl.startswith("pre")
        is_dog_log = not is_pre
        _im = (init_mode or "").strip().lower()
        is_ws_init = _im.startswith("watershed")

        # Fill defaults from result if caller didn't supply spot lists.
        if spots is None and result is not None:
            spots = [{"x": pk.x, "y": pk.y, "snr": pk.snr} for pk in result.models[result.best_n_peaks].peaks]
        if initial_spots is None and result is not None:
            initial_spots = [{"x": pk.x, "y": pk.y, "snr": pk.snr} for pk in result.models[result.best_n_peaks].init_peaks]

        # Spots
        if spots:
            colors = ["magenta", "white", "orange", "cyan", "yellow", "lime", "red", "deepskyblue"]
            for i, pk in enumerate(spots):
                label = f"P{i+1}: S/N={pk.get('snr', 0.0):.1f}"
                # top: absolute
                self.ax_afm.plot(pk["x"], pk["y"], "x", color=colors[i % len(colors)], markersize=10, markeredgewidth=2, label=label)
                if spot_radius_px is not None and spot_radius_px > 0:
                    try:
                        self.ax_afm.add_patch(
                            Circle(
                                (pk["x"], pk["y"]),
                                radius=float(spot_radius_px),
                                fill=False,
                                linewidth=0.8,
                                edgecolor="white",
                                linestyle="-",
                                alpha=0.9,
                            )
                            )
                    except Exception:
                        pass
            self.ax_afm.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )

        if roi_overlay:
            # top: ROI outline in absolute coords
            if roi_overlay.get("shape") == "Ellipse":
                ellipse = Ellipse(
                    (roi_overlay["cx"], roi_overlay["cy"]),
                    width=roi_overlay["rx"] * 2.0,
                    height=roi_overlay["ry"] * 2.0,
                    linewidth=1.5,
                    edgecolor="white",
                    facecolor="none",
                    linestyle="--",
                )
                self.ax_afm.add_patch(ellipse)
            else:
                rect = Rectangle(
                    (roi_overlay["x0"], roi_overlay["y0"]),
                    roi_overlay["w"],
                    roi_overlay["h"],
                    linewidth=1.5,
                    edgecolor="white",
                    facecolor="none",
                    linestyle="--",
                )
                self.ax_afm.add_patch(rect)

            # bottom: plot spots in ROI-local coords (optional)
            # Rules:
            # - Watershed initializer runs on ROI-pre (intensity), so show init spots on ROI-pre.
            # - Otherwise:
            #   - ROI-pre (left): show init spots when detection label is Preprocessed.
            #   - ROI-det (right): show init spots when detection label is DoG/LoG.
            show_init_on_pre = bool(show_roi_spots and initial_spots and (is_pre or is_ws_init))
            show_init_on_det = bool(show_det_spots and initial_spots and is_dog_log and (not is_ws_init))
            if roi_overlay and (show_init_on_pre or show_init_on_det or (show_fit_spots_on_roi_pre and spots)):
                try:
                    # Use result.origin for coordinate conversion if available (accounts for boundary clipping)
                    if result is not None:
                        x0 = float(result.origin[0])
                        y0 = float(result.origin[1])
                    else:
                        # Fallback: clip roi_overlay coordinates to image boundaries
                        h_img, w_img = frame.shape
                        x0_raw = float(roi_overlay.get("x0", 0.0))
                        y0_raw = float(roi_overlay.get("y0", 0.0))
                        x0 = float(max(int(round(x0_raw)), 0))
                        y0 = float(max(int(round(y0_raw)), 0))

                    w = float(roi_overlay.get("w", 0.0))
                    h = float(roi_overlay.get("h", 0.0))

                    # Settings for margin checks; positions excluded by ROI margin are not displayed.
                    margin = 0
                    roi_mask_check = None
                    try:
                        # Get margin value from SpotAnalysisPlugin
                        from PyQt5.QtWidgets import QApplication
                        for widget in QApplication.topLevelWidgets():
                            if hasattr(widget, 'margin_spin'):
                                margin = int(widget.margin_spin.value())
                                break
                    except Exception:
                        pass

                    # Create a mask for elliptical ROI
                    if roi_overlay.get("shape") == "Ellipse":
                        try:
                            cx = float(roi_overlay.get("cx", 0.0))
                            cy = float(roi_overlay.get("cy", 0.0))
                            rx = float(roi_overlay.get("rx", 1.0))
                            ry = float(roi_overlay.get("ry", 1.0))

                            # Create the mask in ROI-local coordinates
                            yy_grid, xx_grid = np.mgrid[0:int(h), 0:int(w)]
                            mask_full = ((xx_grid + x0 - cx) ** 2) / (rx ** 2 + 1e-12) + ((yy_grid + y0 - cy) ** 2) / (ry ** 2 + 1e-12) <= 1.0

                            if margin > 0:
                                roi_mask_check = morphology.erosion(mask_full.astype(bool), morphology.disk(margin))
                            else:
                                roi_mask_check = mask_full.astype(bool)
                        except Exception:
                            pass
                    else:
                        # Rectangle ROI
                        if margin > 0:
                            # For boundary checks with the margin amount
                            pass  # Checked individually later
                    if w > 0 and h > 0:
                        colors = ["magenta", "white", "orange", "cyan", "yellow", "lime", "red", "deepskyblue"]
                        # Debug overlay: final (fit) peaks on ROI-pre image.
                        # This helps compare initializer vs fit result in the same coordinate system.
                        if show_fit_spots_on_roi_pre and spots:
                            for i, pk in enumerate(spots):
                                sx = float(pk.get("x", 0.0))
                                sy = float(pk.get("y", 0.0))
                                rx = sx - x0
                                ry = sy - y0

                                # Margin check to avoid display noise. This is also used for isolating coordinate mismatches, so
                                # outside-ROI points are skipped as-is because they are outside the axes and invisible.
                                if roi_mask_check is not None:
                                    iy, ix = int(round(ry)), int(round(rx))
                                    if 0 <= iy < roi_mask_check.shape[0] and 0 <= ix < roi_mask_check.shape[1]:
                                        if not roi_mask_check[iy, ix]:
                                            continue
                                    else:
                                        continue
                                elif margin > 0:
                                    if rx < margin or ry < margin or rx > w - margin or ry > h - margin:
                                        continue

                                # Do not display outside-ROI points because they are invisible.
                                if rx < 0 or ry < 0 or rx > w or ry > h:
                                    continue

                                self.ax_roi_pre.plot(
                                    rx,
                                    ry,
                                    "o",
                                    markerfacecolor="none",
                                    markeredgecolor=colors[i % len(colors)],
                                    markersize=10,
                                    markeredgewidth=2.0,
                                    linestyle="None",
                                    alpha=0.95,
                                )
                                if spot_radius_px is not None and spot_radius_px > 0:
                                    self.ax_roi_pre.add_patch(
                                        Circle(
                                            (rx, ry),
                                            radius=float(spot_radius_px),
                                            fill=False,
                                            linewidth=0.8,
                                            edgecolor=colors[i % len(colors)],
                                            linestyle="-",
                                            alpha=0.85,
                                        )
                                    )
                        if show_init_on_pre and initial_spots:
                            for i, pk in enumerate(initial_spots):
                                sx = float(pk.get("x", 0.0))
                                sy = float(pk.get("y", 0.0))
                                if sx < x0 or sy < y0 or sx > x0 + w or sy > y0 + h:
                                    continue
                                rx = sx - x0
                                ry = sy - y0

                                # Margin check
                                if roi_mask_check is not None:
                                    # Ellipse ROI: check by mask
                                    iy, ix = int(round(ry)), int(round(rx))
                                    if 0 <= iy < roi_mask_check.shape[0] and 0 <= ix < roi_mask_check.shape[1]:
                                        if not roi_mask_check[iy, ix]:
                                            continue  # Skip because it is inside the margin range
                                    else:
                                        continue
                                elif margin > 0:
                                    # Rectangle ROI: check by distance from boundary
                                    if rx < margin or ry < margin or rx > w - margin or ry > h - margin:
                                        continue  # Skip because it is inside the margin range
                                self.ax_roi_pre.plot(
                                    rx, ry, "x", color=colors[i % len(colors)], markersize=9, markeredgewidth=2
                                )
                                if spot_radius_px is not None and spot_radius_px > 0:
                                    self.ax_roi_pre.add_patch(
                                        Circle(
                                            (rx, ry),
                                            radius=float(spot_radius_px),
                                            fill=False,
                                            linewidth=0.8,
                                            edgecolor="white",
                                            linestyle="-",
                                            alpha=0.9,
                                        )
                                    )
                        if show_init_on_det and initial_spots:
                            for i, pk in enumerate(initial_spots):
                                sx = float(pk.get("x", 0.0))
                                sy = float(pk.get("y", 0.0))
                                if sx < x0 or sy < y0 or sx > x0 + w or sy > y0 + h:
                                    continue
                                rx = sx - x0
                                ry = sy - y0

                                # Margin check
                                if roi_mask_check is not None:
                                    # Ellipse ROI: check by mask
                                    iy, ix = int(round(ry)), int(round(rx))
                                    if 0 <= iy < roi_mask_check.shape[0] and 0 <= ix < roi_mask_check.shape[1]:
                                        if not roi_mask_check[iy, ix]:
                                            continue  # Skip because it is inside the margin range
                                    else:
                                        continue
                                elif margin > 0:
                                    # Rectangle ROI: check by distance from boundary
                                    if rx < margin or ry < margin or rx > w - margin or ry > h - margin:
                                        continue  # Skip because it is inside the margin range
                                self.ax_roi_det.plot(
                                    rx, ry, "x", color=colors[i % len(colors)], markersize=9, markeredgewidth=2
                                )
                                if spot_radius_px is not None and spot_radius_px > 0:
                                    self.ax_roi_det.add_patch(
                                        Circle(
                                            (rx, ry),
                                            radius=float(spot_radius_px),
                                            fill=False,
                                            linewidth=0.8,
                                            edgecolor="white",
                                            linestyle="-",
                                            alpha=0.9,
                                        )
                                    )
                except Exception:
                    pass

        if result is not None:
            self.ax_afm.set_title(f"AFM - {result.best_n_peaks} peaks ({result.criterion.upper()})")
        elif spots is not None:
            # Analysis has run but no peaks survived post-filters (strict ROI margin, etc.)
            self.ax_afm.set_title("AFM - 0 peaks")
        else:
            self.ax_afm.set_title("AFM (no analysis yet)")
        self.ax_roi_pre.set_title("ROI (preprocessed)")
        self.ax_roi_det.set_title(det_label or "ROI (LoG/DoG)")

        self.ax_afm.set_xlabel("X (pixel)")
        self.ax_afm.set_ylabel("Y (pixel)")
        for ax in (self.ax_roi_pre, self.ax_roi_det):
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
        self.canvas.draw()

    def _safe_remove_colorbar(self):
        """Safely remove the colorbar to avoid Matplotlib KeyError issues"""
        try:
            if self.cbar:
                self.cbar.remove()
        except Exception:
            pass
        self.cbar = None

    def _on_press(self, event):
        if self.edit_handler is None:
            return
        # While Cmd/Ctrl is pressed, pause ROI selector and let spot editing handle drag events.
        is_mod = False
        try:
            gui_ev = getattr(event, "guiEvent", None)
            if gui_ev is not None and hasattr(gui_ev, "modifiers"):
                mods = gui_ev.modifiers()
                is_mod = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
        except Exception:
            is_mod = False
        if not is_mod:
            try:
                mods = QtWidgets.QApplication.keyboardModifiers()
                is_mod = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
            except Exception:
                pass
        if is_mod and self.selector is not None:
            try:
                self.selector.set_active(False)
                self._roi_selector_paused = True
            except Exception:
                pass
        self.edit_handler(event, "press")

    def _on_motion(self, event):
        if self.edit_handler is None:
            return
        self.edit_handler(event, "move")

    def _on_release(self, event):
        if self.edit_handler is None:
            return
        self.edit_handler(event, "release")
        # Restore the paused ROI selector
        if self._roi_selector_paused and self.selector is not None:
            try:
                self.selector.set_active(True)
            except Exception:
                pass
        self._roi_selector_paused = False


class SpotAnalysisWindow(QtWidgets.QWidget):
    """
    Simple on-demand UI launched from the pyNuD main window.
    Runs SpotAnalysis on the current frame of the file selected in FileList.
    """

    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Spot Analysis (AIC/BIC)")
        self.setMinimumWidth(420)
        self.spot_analyzer = SpotAnalysis()
        self.viz_window = None  # ROI visualization window
        self.full_viz_window = None  # Full-image overlay window
        self.last_frame = None
        self.last_result = None
        self.manual_roi = None  # dict: {"shape": str, "x0":..., "y0":..., "w":..., "h":..., "cx":..., "cy":..., "rx":..., "ry":...}
        self.roi_by_frame: Dict[int, Dict[str, float]] = {}
        self.spots_by_frame: Dict[int, List[Dict[str, float]]] = {}
        self.centroid_reference_by_frame: Dict[int, List[Dict[str, float]]] = {}
        self.initial_spots_by_frame: Dict[int, List[Dict[str, float]]] = {}
        self.preserve_existing_spots_on_auto = False
        self.export_dir = None
        self._dragging = False
        self._drag_index = None
        self._auto_busy = False
        self._analysis_signature_by_frame: Dict[int, Tuple] = {}
        self._reanalysis_timer = QtCore.QTimer(self)
        self._reanalysis_timer.setSingleShot(True)
        self._reanalysis_timer.timeout.connect(self._reanalyze_current_frame_debounced)
        self._build_ui()
        self._refresh_selection_label()
        self._connect_frame_signal()
        self.show_full_image_view()
        self._update_frame_label()

    def closeEvent(self, event):
        """Close visualization windows when this window closes"""
        if self.viz_window:
            self.viz_window.close()
        super().closeEvent(event)

    @staticmethod
    def _help_html(body: str) -> str:
        return f"<html><head>{HELP_CSS}</head><body>{body}</body></html>"

    def _show_help(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{PLUGIN_NAME} Help")
        dialog.resize(900, 720)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(QtWidgets.QLabel("Language / 言語:"))
        btn_ja = QtWidgets.QPushButton("日本語", dialog)
        btn_en = QtWidgets.QPushButton("English", dialog)
        btn_ja.setCheckable(True)
        btn_en.setCheckable(True)
        lang_group = QtWidgets.QButtonGroup(dialog)
        lang_group.addButton(btn_ja)
        lang_group.addButton(btn_en)
        lang_group.setExclusive(True)
        selected_style = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        normal_style = "QPushButton { background-color: #e5e5e5; color: black; }"
        lang_row.addWidget(btn_ja)
        lang_row.addWidget(btn_en)
        lang_row.addStretch(1)
        layout.addLayout(lang_row)

        heading = QtWidgets.QLabel("Spot Analysis Help")
        heading_font = heading.font()
        heading_font.setPointSize(max(heading_font.pointSize() + 3, 14))
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        summary = QtWidgets.QLabel("")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        layout.addWidget(tabs, stretch=1)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        def set_lang(use_ja: bool) -> None:
            btn_ja.setChecked(use_ja)
            btn_en.setChecked(not use_ja)
            btn_ja.setStyleSheet(selected_style if use_ja else normal_style)
            btn_en.setStyleSheet(selected_style if not use_ja else normal_style)
            summary.setText(
                "ROIを指定して輝点を検出し、AIC/BICでピーク数を判定し、"
                "CSVへ保存するまでの操作ガイドです。"
                if use_ja
                else "Guide for selecting an ROI, detecting bright spots, choosing peak counts with AIC/BIC, "
                "editing results, and exporting CSV files."
            )
            tabs.clear()
            for title, html_text in (HELP_TABS_JA if use_ja else HELP_TABS_EN):
                browser = QtWidgets.QTextBrowser()
                browser.setOpenExternalLinks(False)
                browser.setHtml(self._help_html(html_text))
                browser.setMinimumWidth(720)
                tabs.addTab(browser, title)

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        set_lang(False)

        dialog.exec_()

    def _build_ui(self) -> None:
        outer_layout = QtWidgets.QVBoxLayout(self)

        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        help_menu = menu_bar.addMenu("Help")
        manual_action = help_menu.addAction("Manual")
        manual_action.setStatusTip("Open Spot Analysis help.")
        manual_action.triggered.connect(self._show_help)
        outer_layout.setMenuBar(menu_bar)

        def _setup_grid(grid: QtWidgets.QGridLayout) -> None:
            # 0: label, 1: input, 2: expanding spacer column
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 0)
            grid.setColumnStretch(2, 1)
            try:
                grid.setHorizontalSpacing(12)
            except Exception:
                pass

        def _add_right_spacer(grid: QtWidgets.QGridLayout, row_span: int) -> None:
            try:
                grid.addItem(
                    QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum),
                    0,
                    2,
                    max(1, int(row_span)),
                    1,
                )
            except Exception:
                pass

        # --- Top buttons (always visible) ---
        top_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        self.help_btn = QtWidgets.QPushButton("Help")
        self.help_btn.setToolTip("Open the Spot Analysis operation guide.")
        self.help_btn.clicked.connect(self._show_help)

        self.refit_manual_btn = QtWidgets.QPushButton("Move Spots to Radius Centroid")
        self.refit_manual_btn.setToolTip("Move each spot to the intensity centroid inside its spot-radius circle for the current frame only.")
        self.refit_manual_btn.clicked.connect(self.refit_from_manual_spots)
        self.refit_manual_btn.setEnabled(False)
        self.auto_centroid_check = QtWidgets.QCheckBox("Auto Apply")
        self.auto_centroid_check.setChecked(False)
        self.auto_centroid_check.setToolTip(
            "When ON, automatically moves each spot to the intensity centroid inside its spot radius after analysis."
        )

        self.run_all_btn = QtWidgets.QPushButton("All Frames Analysis")
        self.run_all_btn.clicked.connect(self.run_analysis_all_frames)
        self.run_all_btn.setEnabled(False)

        self.run_all_enable_check = QtWidgets.QCheckBox("All Frames Analysis")
        self.run_all_enable_check.setChecked(False)
        self.run_all_enable_check.setToolTip("The All Frames Analysis button is enabled only when this checkbox is ON.")
        self.run_all_enable_check.toggled.connect(self._sync_run_buttons_enabled)

        self.prev_frame_btn = QtWidgets.QPushButton("◀")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        self.next_frame_btn = QtWidgets.QPushButton("▶")
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.frame_label = QtWidgets.QLabel("Frame: - / -")

        # --- 1st row: Run button + Frame navigation ---
        top_row.addWidget(self.run_btn)
        top_row.addWidget(self.prev_frame_btn)
        top_row.addWidget(self.next_frame_btn)
        top_row.addWidget(self.frame_label)
        top_row.addWidget(self.help_btn)
        top_row.addStretch(1)
        outer_layout.addLayout(top_row)

        # --- 2nd row: Manual spot centroid placement ---
        refit_row = QtWidgets.QHBoxLayout()
        refit_row.addWidget(self.refit_manual_btn)
        refit_row.addWidget(self.auto_centroid_check)
        refit_row.addStretch(1)
        outer_layout.addLayout(refit_row)

        # --- 3rd row: Run all frames + enable ---
        run_all_row = QtWidgets.QHBoxLayout()
        run_all_row.addWidget(self.run_all_btn)
        run_all_row.addWidget(self.run_all_enable_check)
        run_all_row.addStretch(1)
        outer_layout.addLayout(run_all_row)

        # --- Scroll area ---
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer_layout.addWidget(self._scroll)
        content = QtWidgets.QWidget()
        self._scroll.setWidget(content)
        layout = QtWidgets.QHBoxLayout(content)

        # --- Form widget (fixed width, left-aligned) ---
        form_widget = QtWidgets.QWidget()
        form_widget.setMaximumWidth(440)
        form_layout = QtWidgets.QVBoxLayout(form_widget)

        # --- Groups ---
        basic_group = QtWidgets.QGroupBox("ROI / Basic")
        basic_grid = QtWidgets.QGridLayout(basic_group)
        _setup_grid(basic_grid)
        r = 0
        roi_shape_label = QtWidgets.QLabel("ROI Shape")
        roi_shape_label.setToolTip(
            "Analysis ROI shape:\n"
            "- Rectangle: rectangular ROI\n"
            "- Ellipse (Circle): elliptical/circular ROI; pixels outside the boundary are excluded"
        )
        basic_grid.addWidget(roi_shape_label, r, 0)
        self.roi_shape_combo = QtWidgets.QComboBox()
        self.roi_shape_combo.addItems(["Rectangle", "Ellipse (Circle)"])
        self.roi_shape_combo.setToolTip("Select the ROI shape.")
        self.roi_shape_combo.setMaximumWidth(150)
        self.roi_shape_combo.currentTextChanged.connect(self._on_roi_shape_changed)
        basic_grid.addWidget(self.roi_shape_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        auto_analyze_label = QtWidgets.QLabel("Auto Analysis")
        auto_analyze_label.setToolTip(
            "Automatically run analysis when changing frames:\n"
            "when enabled, peak detection and fitting run automatically\n"
            "when switching to a frame that has an ROI."
        )
        basic_grid.addWidget(auto_analyze_label, r, 0)
        self.auto_analyze_check = QtWidgets.QCheckBox("Auto-analyze on frame change")
        self.auto_analyze_check.setChecked(True)  # Enabled by default
        self.auto_analyze_check.setToolTip("Automatically analyze frames that have an ROI.")
        basic_grid.addWidget(self.auto_analyze_check, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        criterion_label = QtWidgets.QLabel("Information Criterion")
        criterion_label.setToolTip(
            "Information criterion used to select the optimal number of peaks:\n"
            "- AIC (Akaike Information Criterion): emphasizes predictive accuracy\n"
            "- BIC (Bayesian Information Criterion): emphasizes model simplicity and is more conservative\n"
            "BIC generally suppresses overfitting more strongly."
        )
        basic_grid.addWidget(criterion_label, r, 0)
        self.criterion_combo = QtWidgets.QComboBox()
        self.criterion_combo.addItems(["AIC", "BIC"])
        self.criterion_combo.setMaximumWidth(100)
        basic_grid.addWidget(self.criterion_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # min/max peaks (paired)
        self.min_peaks_spin = QtWidgets.QSpinBox()
        self.min_peaks_spin.setRange(1, 6)
        self.min_peaks_spin.setValue(1)
        self.min_peaks_spin.setMaximumWidth(70)
        self.max_peaks_spin = QtWidgets.QSpinBox()
        self.max_peaks_spin.setRange(1, 6)
        self.max_peaks_spin.setValue(3)
        self.max_peaks_spin.setMaximumWidth(70)
        peaks_row = QtWidgets.QHBoxLayout()
        peaks_row.addWidget(QtWidgets.QLabel("Min"))
        peaks_row.addWidget(self.min_peaks_spin)
        peaks_row.addSpacing(8)
        peaks_row.addWidget(QtWidgets.QLabel("Max"))
        peaks_row.addWidget(self.max_peaks_spin)
        peaks_label = QtWidgets.QLabel("Peak Count")
        peaks_label.setToolTip(
            "Range of peak counts tested during fitting.\n"
            "Also used as the upper limit for Peak-type initial seed candidates.\n"
            "The optimal model is selected by AIC/BIC."
        )
        basic_grid.addWidget(peaks_label, r, 0)
        basic_grid.addLayout(peaks_row, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        _add_right_spacer(basic_grid, r)
        form_layout.addWidget(basic_group)

        preprocess_group = QtWidgets.QGroupBox("Preprocessing (inside ROI)")
        pre_grid = QtWidgets.QGridLayout(preprocess_group)
        _setup_grid(pre_grid)
        r = 0
        self.median_check = QtWidgets.QCheckBox("Median")
        self.median_check.setChecked(True)  # Enabled by default
        self.median_size_spin = QtWidgets.QSpinBox()
        self.median_size_spin.setRange(1, 31)
        self.median_size_spin.setSingleStep(2)
        self.median_size_spin.setValue(int(getattr(self.spot_analyzer, "median_size", 3)))
        self.median_size_spin.setToolTip("Median kernel size; odd values are recommended.")
        self.median_size_spin.setMaximumWidth(80)
        median_row = QtWidgets.QHBoxLayout()
        median_row.addWidget(self.median_check)
        median_row.addSpacing(8)
        median_k_label = QtWidgets.QLabel("k")
        median_k_label.setToolTip("Kernel size of the median filter. Removes noise while preserving spots.")
        median_row.addWidget(median_k_label)
        median_row.addWidget(self.median_size_spin)
        pre_grid.addLayout(median_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1

        self.open_check = QtWidgets.QCheckBox("Open")
        self.open_radius_spin = QtWidgets.QSpinBox()
        self.open_radius_spin.setRange(0, 20)
        self.open_radius_spin.setValue(int(getattr(self.spot_analyzer, "open_radius", 1)))
        self.open_radius_spin.setToolTip("Opening radius (px)")
        self.open_radius_spin.setMaximumWidth(80)
        open_row = QtWidgets.QHBoxLayout()
        open_row.addWidget(self.open_check)
        open_row.addSpacing(8)
        open_r_label = QtWidgets.QLabel("r(px)")
        open_r_label.setToolTip("Radius for morphological opening. Removes small protrusions and noise.")
        open_row.addWidget(open_r_label)
        open_row.addWidget(self.open_radius_spin)
        pre_grid.addLayout(open_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(pre_grid, r)
        form_layout.addWidget(preprocess_group)

        detect_group = QtWidgets.QGroupBox("Detection (LoG / DoG / Pre)")
        det_grid = QtWidgets.QGridLayout(detect_group)
        _setup_grid(det_grid)
        r = 0
        detection_mode_label = QtWidgets.QLabel("Method")
        detection_mode_label.setToolTip(
            "Initial peak-position detection method:\n"
            "- DoG: emphasizes edges/blobs using Difference of Gaussians (sigma high - sigma low)\n"
            "- LoG: detects blobs using Laplacian of Gaussian with one sigma\n"
            "- Preprocessed: uses the preprocessed image directly (median/morphological opening)"
        )
        det_grid.addWidget(detection_mode_label, r, 0)
        self.detection_mode_combo = QtWidgets.QComboBox()
        self.detection_mode_combo.addItems(["DoG", "LoG", "Preprocessed"])
        self.detection_mode_combo.setCurrentText("LoG")
        self.detection_mode_combo.setMaximumWidth(100)
        self.detection_mode_combo.currentTextChanged.connect(self._update_detection_ui_enabled)
        det_grid.addWidget(self.detection_mode_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # DoG parameters (paired)
        self.bandpass_low_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_low_spin.setRange(0.1, 20.0)
        self.bandpass_low_spin.setSingleStep(0.1)
        self.bandpass_low_spin.setValue(self.spot_analyzer.bandpass_low_sigma)
        self.bandpass_low_spin.setMaximumWidth(90)
        self.bandpass_high_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_high_spin.setRange(0.1, 50.0)
        self.bandpass_high_spin.setSingleStep(0.1)
        self.bandpass_high_spin.setValue(self.spot_analyzer.bandpass_high_sigma)
        self.bandpass_high_spin.setMaximumWidth(90)
        dog_row = QtWidgets.QHBoxLayout()
        dog_sigma_low_label = QtWidgets.QLabel("σ low")
        dog_sigma_low_label.setToolTip("Gaussian blur radius on the low-sigma side of DoG. Smaller values support smaller spots.")
        dog_row.addWidget(dog_sigma_low_label)
        dog_row.addWidget(self.bandpass_low_spin)
        dog_row.addSpacing(8)
        dog_sigma_high_label = QtWidgets.QLabel("σ high")
        dog_sigma_high_label.setToolTip("Gaussian blur radius on the high-sigma side of DoG. Removes larger structures.")
        dog_row.addWidget(dog_sigma_high_label)
        dog_row.addWidget(self.bandpass_high_spin)
        det_grid.addLayout(dog_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1

        # LoG parameter
        self.log_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.log_sigma_spin.setRange(0.1, 50.0)
        self.log_sigma_spin.setSingleStep(0.1)
        self.log_sigma_spin.setValue(float(getattr(self.spot_analyzer, "log_sigma", 1.6)))
        self.log_sigma_spin.setMaximumWidth(90)
        log_row = QtWidgets.QHBoxLayout()
        log_sigma_label = QtWidgets.QLabel("σ")
        log_sigma_label.setToolTip("Gaussian blur radius for LoG. Tune it to the expected spot size.")
        log_row.addWidget(log_sigma_label)
        log_row.addWidget(self.log_sigma_spin)
        det_grid.addLayout(log_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(det_grid, r)

        form_layout.addWidget(detect_group)

        # NOTE: This group contains ONLY initializer (seed detection) related controls.
        fit_group = QtWidgets.QGroupBox("Initialization (initial positions)")
        fit_grid = QtWidgets.QGridLayout(fit_group)
        _setup_grid(fit_grid)
        r = 0

        # ROI margin (used in initial seed selection AND result filtering)
        self.margin_spin = QtWidgets.QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setValue(self.spot_analyzer.margin)
        self.margin_spin.setMaximumWidth(70)
        self.margin_spin.setToolTip("Exclusion distance from ROI boundary (px)")
        margin_label = QtWidgets.QLabel("ROI Margin (px)")
        margin_label.setToolTip(
            "Exclusion distance from the ROI boundary in pixels:\n"
            "Used both for initial candidate selection and final result filtering.\n"
            "Peaks within this distance from the ROI boundary are excluded."
        )
        fit_grid.addWidget(margin_label, r, 0)
        fit_grid.addWidget(self.margin_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Peak spacing (used in Peak seed NMS; also a sensible global spacing hint)
        self.peak_min_distance_spin = QtWidgets.QSpinBox()
        self.peak_min_distance_spin.setRange(1, 50)
        self.peak_min_distance_spin.setValue(self.spot_analyzer.peak_min_distance)
        self.peak_min_distance_spin.setMaximumWidth(70)
        peak_min_distance_label = QtWidgets.QLabel("Peak Spacing (px)")
        peak_min_distance_label.setToolTip(
            "Minimum distance between peaks in pixels:\n"
            "For Peak-type initial candidates, only the strongest candidate within this distance is kept (NMS).\n"
            "Prevents duplicate detection of close peaks."
        )
        fit_grid.addWidget(peak_min_distance_label, r, 0)
        fit_grid.addWidget(self.peak_min_distance_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # initializer mode
        init_mode_label = QtWidgets.QLabel("Initial Candidates")
        init_mode_label.setToolTip(
            "Initial peak-candidate generation method for Gaussian fitting:\n"
            "- Watershed: robust for complex images using adaptive h-maxima (recommended)\n"
            "- Blob DoH: efficient and well suited for circular spot detection\n"
            "- Peak: simple local-maximum detection"
        )
        fit_grid.addWidget(init_mode_label, r, 0)
        self.init_mode_combo = QtWidgets.QComboBox()
        self.init_mode_combo.addItems([
            "Watershed (Recommended)",
            "Blob DoH (Fast)",
            "Peak"
        ])
        self.init_mode_combo.setCurrentText("Peak")
        self.init_mode_combo.setMaximumWidth(180)
        self.init_mode_combo.setToolTip(
            "Initial peak-candidate generation method for Gaussian fitting:\n\n"
            "• Watershed (Recommended)\n"
            "  Robust for complex images using adaptive h-maxima\n\n"
            "• Blob DoH (Fast)\n"
            "  Efficient and well suited for circular spot detection\n\n"
            "• Peak\n"
            "  Simple local-maximum detection"
        )
        self.init_mode_combo.currentTextChanged.connect(self._update_init_mode_ui_enabled)
        fit_grid.addWidget(self.init_mode_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Subpixel refinement option
        self.subpixel_check = QtWidgets.QCheckBox("Subpixel Refinement")
        self.subpixel_check.setChecked(False)
        self.subpixel_check.setToolTip("Refine positions to subpixel accuracy in Peak mode.")
        fit_grid.addWidget(self.subpixel_check, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.watershed_h_rel_spin = QtWidgets.QDoubleSpinBox()
        self.watershed_h_rel_spin.setRange(0.0, 1.0)
        self.watershed_h_rel_spin.setSingleStep(0.01)
        self.watershed_h_rel_spin.setDecimals(3)
        self.watershed_h_rel_spin.setValue(float(getattr(self.spot_analyzer, "watershed_h_rel", 0.05)))
        self.watershed_h_rel_spin.setMaximumWidth(90)
        self.watershed_h_rel_spin.setToolTip("Relative h-maxima height. Larger values suppress over-segmentation (0-1).")
        ws_h_label = QtWidgets.QLabel("WS h(rel)")
        ws_h_label.setToolTip(
            "Relative height for Watershed h-maxima (0-1):\n"
            "Sets noise suppression strength as a fraction of the image intensity range.\n"
            "Larger values ignore smaller local maxima and suppress over-segmentation.\n"
            "When adaptive h-maxima is enabled, the noise level is also considered."
        )
        fit_grid.addWidget(ws_h_label, r, 0)
        fit_grid.addWidget(self.watershed_h_rel_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.watershed_adaptive_h_check = QtWidgets.QCheckBox("Adaptive h-maxima")
        self.watershed_adaptive_h_check.setChecked(bool(getattr(self.spot_analyzer, "watershed_adaptive_h", True)))
        self.watershed_adaptive_h_check.setToolTip("Automatically adjusts the h value based on the noise level.")
        fit_grid.addWidget(self.watershed_adaptive_h_check, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # blob_doh params
        self.blob_doh_min_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.blob_doh_min_sigma_spin.setRange(0.2, 200.0)
        self.blob_doh_min_sigma_spin.setSingleStep(0.5)
        self.blob_doh_min_sigma_spin.setValue(float(getattr(self.spot_analyzer, "blob_doh_min_sigma", 3.0)))
        self.blob_doh_min_sigma_spin.setMaximumWidth(90)
        self.blob_doh_max_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.blob_doh_max_sigma_spin.setRange(0.2, 400.0)
        self.blob_doh_max_sigma_spin.setSingleStep(0.5)
        self.blob_doh_max_sigma_spin.setValue(float(getattr(self.spot_analyzer, "blob_doh_max_sigma", 20.0)))
        self.blob_doh_max_sigma_spin.setMaximumWidth(90)
        blob_doh_sigma_row = QtWidgets.QHBoxLayout()
        doh_min_sigma_label = QtWidgets.QLabel("minσ")
        doh_min_sigma_label.setToolTip("Minimum sigma for DoH (Determinant of Hessian) detection")
        blob_doh_sigma_row.addWidget(doh_min_sigma_label)
        blob_doh_sigma_row.addWidget(self.blob_doh_min_sigma_spin)
        blob_doh_sigma_row.addSpacing(8)
        doh_max_sigma_label = QtWidgets.QLabel("maxσ")
        doh_max_sigma_label.setToolTip("Maximum sigma for DoH (Determinant of Hessian) detection")
        blob_doh_sigma_row.addWidget(doh_max_sigma_label)
        blob_doh_sigma_row.addWidget(self.blob_doh_max_sigma_spin)
        doh_sigma_range_label = QtWidgets.QLabel("DoH Sigma Range")
        doh_sigma_range_label.setToolTip(
            "Sigma range for DoH (Determinant of Hessian) blob detection:\n"
            "This fast blob detector uses the determinant of the Hessian matrix.\n"
            "It is more efficient than LoG and suitable for circular spot detection."
        )
        fit_grid.addWidget(doh_sigma_range_label, r, 0)
        fit_grid.addLayout(blob_doh_sigma_row, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.blob_doh_threshold_rel_spin = QtWidgets.QDoubleSpinBox()
        self.blob_doh_threshold_rel_spin.setRange(0.0, 1.0)
        self.blob_doh_threshold_rel_spin.setSingleStep(0.001)
        self.blob_doh_threshold_rel_spin.setDecimals(3)
        self.blob_doh_threshold_rel_spin.setValue(float(getattr(self.spot_analyzer, "blob_doh_threshold_rel", 0.01)))
        self.blob_doh_threshold_rel_spin.setMaximumWidth(90)
        doh_thr_label = QtWidgets.QLabel("DoH thr(rel)")
        doh_thr_label.setToolTip(
            "Relative threshold for DoH detection (0-1):\n"
            "Sets the threshold as a fraction of the maximum DoH response.\n"
            "Lower values detect more blobs."
        )
        fit_grid.addWidget(doh_thr_label, r, 0)
        fit_grid.addWidget(self.blob_doh_threshold_rel_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.localmax_thr_spin = QtWidgets.QDoubleSpinBox()
        self.localmax_thr_spin.setRange(0.0, 1.0)
        self.localmax_thr_spin.setSingleStep(0.01)
        self.localmax_thr_spin.setDecimals(3)
        self.localmax_thr_spin.setValue(float(getattr(self.spot_analyzer, "localmax_threshold_rel", 0.05)))
        self.localmax_thr_spin.setMaximumWidth(90)
        self.localmax_thr_spin.setToolTip("Threshold for peak_local_max, relative to the maximum value in the detection image (LoG/DoG) (0-1).")
        localmax_thr_label = QtWidgets.QLabel("localmax thr(rel)")
        localmax_thr_label.setToolTip(
            "Relative peak-detection threshold in the detection image (LoG/DoG) (0-1):\n"
            "Sets the threshold as a fraction of the maximum detection-image value.\n"
            "Example: maximum=100, thr=0.05 -> detect peaks with values >= 5"
        )
        # NOTE: Peak initialization is currently NMS-based, so this threshold is effectively unused.
        # Keep the widget for backward compatibility (save/restore and internal references), but hide it from the UI.
        localmax_thr_label.setVisible(False)
        self.localmax_thr_spin.setVisible(False)
        fit_grid.addWidget(localmax_thr_label, r, 0)
        fit_grid.addWidget(self.localmax_thr_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.localmax_snr_spin = QtWidgets.QDoubleSpinBox()
        self.localmax_snr_spin.setRange(0.0, 50.0)
        self.localmax_snr_spin.setSingleStep(0.1)
        self.localmax_snr_spin.setDecimals(2)
        self.localmax_snr_spin.setValue(float(getattr(self.spot_analyzer, "localmax_threshold_snr", 0.0)))
        self.localmax_snr_spin.setMaximumWidth(90)
        self.localmax_snr_spin.setToolTip("Noise-based threshold for the detection image using MAD. 0 disables it.")
        localmax_snr_label = QtWidgets.QLabel("localmax thr(SNR)")
        localmax_snr_label.setToolTip(
            "Noise-based SNR threshold for the detection image:\n"
            "threshold = thr(SNR) x noise level (MAD)\n"
            "When used with thr(rel), the larger threshold is applied.\n"
            "Set to 0 to disable. Use this when you only want peaks that stand above noise."
        )
        localmax_snr_label.setVisible(False)
        self.localmax_snr_spin.setVisible(False)
        fit_grid.addWidget(localmax_snr_label, r, 0)
        fit_grid.addWidget(self.localmax_snr_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(fit_grid, r)

        form_layout.addWidget(fit_group)

        # --- Fit (Gaussian model) ---
        fit_model_group = QtWidgets.QGroupBox("Fit (Gaussian Model)")
        fit_model_grid = QtWidgets.QGridLayout(fit_model_group)
        _setup_grid(fit_model_grid)
        r = 0

        self.fit_enabled_check = QtWidgets.QCheckBox("Enable Gaussian Fit")
        self.fit_enabled_check.setChecked(bool(getattr(self.spot_analyzer, "fit_enabled", True)))
        self.fit_enabled_check.setToolTip(
            "When OFF, peak coordinates found by initial-position search are used directly as the final result.\n"
            "Coordinate optimization is skipped, and AIC/BIC are evaluated using the initial model."
        )
        self.fit_enabled_check.toggled.connect(self._update_fit_ui_enabled)
        fit_model_grid.addWidget(self.fit_enabled_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.initial_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.initial_sigma_spin.setRange(0.1, 20.0)
        self.initial_sigma_spin.setSingleStep(0.1)
        self.initial_sigma_spin.setValue(self.spot_analyzer.initial_sigma)
        self.initial_sigma_spin.setMaximumWidth(90)
        initial_sigma_label = QtWidgets.QLabel("Initial Sigma")
        initial_sigma_label.setToolTip(
            "Initial sigma value for Gaussian fitting:\n"
            "Used as the initial sigma for peak candidates.\n"
            "Set it according to the typical spot size."
        )
        fit_model_grid.addWidget(initial_sigma_label, r, 0)
        fit_model_grid.addWidget(self.initial_sigma_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # sigma bounds (paired)
        self.sigma_min_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_min_spin.setRange(0.05, 50.0)
        self.sigma_min_spin.setSingleStep(0.05)
        self.sigma_min_spin.setValue(self.spot_analyzer.sigma_bounds[0])
        self.sigma_min_spin.setMaximumWidth(90)
        self.sigma_max_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_max_spin.setRange(0.1, 80.0)
        self.sigma_max_spin.setSingleStep(0.1)
        self.sigma_max_spin.setValue(self.spot_analyzer.sigma_bounds[1])
        self.sigma_max_spin.setMaximumWidth(90)
        sigma_row = QtWidgets.QHBoxLayout()
        sigma_lower_label = QtWidgets.QLabel("Lower")
        sigma_lower_label.setToolTip("Minimum sigma during fitting. The fit will not converge below this value.")
        sigma_row.addWidget(sigma_lower_label)
        sigma_row.addWidget(self.sigma_min_spin)
        sigma_row.addSpacing(8)
        sigma_upper_label = QtWidgets.QLabel("Upper")
        sigma_upper_label.setToolTip("Maximum sigma during fitting. The fit will not converge above this value.")
        sigma_row.addWidget(sigma_upper_label)
        sigma_row.addWidget(self.sigma_max_spin)
        sigma_bounds_label = QtWidgets.QLabel("σ")
        sigma_bounds_label.setToolTip(
            "Sigma constraint range during Gaussian fitting:\n"
            "Sigma is constrained to this range during fitting.\n"
            "Set it according to the expected variation in spot size."
        )
        fit_model_grid.addWidget(sigma_bounds_label, r, 0)
        fit_model_grid.addLayout(sigma_row, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(fit_model_grid, r)

        form_layout.addWidget(fit_model_group)

        filter_group = QtWidgets.QGroupBox("Result Filters")
        filter_grid = QtWidgets.QGridLayout(filter_group)
        _setup_grid(filter_grid)
        r = 0

        # --- Post-processing / filters ---
        self.snr_spin = QtWidgets.QDoubleSpinBox()
        self.snr_spin.setRange(0.1, 50.0)
        self.snr_spin.setSingleStep(0.1)
        self.snr_spin.setValue(self.spot_analyzer.snr_threshold)
        self.snr_spin.setMaximumWidth(90)
        snr_label = QtWidgets.QLabel("S/N Threshold (output filter)")
        snr_label.setToolTip(
            "Minimum S/N ratio (Signal-to-Noise Ratio) for peaks included in final results:\n"
            "S/N = amplitude / noise level\n"
            "Peaks below this value are excluded from final results.\n"
            "Use a higher value for noisy images."
        )
        filter_grid.addWidget(snr_label, r, 0)
        filter_grid.addWidget(self.snr_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # precheck (fit-result validation on roi_pre)
        self.precheck_radius_spin = QtWidgets.QSpinBox()
        self.precheck_radius_spin.setRange(0, 20)
        self.precheck_radius_spin.setValue(int(getattr(self.spot_analyzer, "precheck_radius_px", 2)))
        self.precheck_radius_spin.setMaximumWidth(70)
        self.precheck_radius_spin.setToolTip("Local-peak validation radius on the original ROI image. 0 disables it.")
        precheck_r_label = QtWidgets.QLabel("precheck r(px)")
        precheck_r_label.setToolTip(
            "Local-maximum validation radius on the original image, in pixels:\n"
            "Checks whether each peak candidate is the maximum inside an r(px) window.\n"
            "If it is more than 10% below the window maximum, the peak is excluded.\n"
            "Set to 0 to disable. This removes peaks detected in the detection image that are not local maxima in the original image."
        )
        filter_grid.addWidget(precheck_r_label, r, 0)
        filter_grid.addWidget(self.precheck_radius_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.precheck_kmad_spin = QtWidgets.QDoubleSpinBox()
        self.precheck_kmad_spin.setRange(0.0, 10.0)
        self.precheck_kmad_spin.setSingleStep(0.1)
        self.precheck_kmad_spin.setDecimals(2)
        self.precheck_kmad_spin.setValue(float(getattr(self.spot_analyzer, "precheck_kmad", 1.0)))
        self.precheck_kmad_spin.setMaximumWidth(90)
        self.precheck_kmad_spin.setToolTip("Exclude peaks below median + K*MAD in the original ROI image. 0 disables it.")
        precheck_kmad_label = QtWidgets.QLabel("precheck K(MAD)")
        precheck_kmad_label.setToolTip(
            "Noise-based threshold coefficient for the original image:\n"
            "Calculates median + K*MAD in a window around each peak.\n"
            "If the peak intensity is below this value, the peak is excluded.\n"
            "MAD = Median Absolute Deviation, a robust variance estimate.\n"
            "Set to 0 to disable. Use this when you only want peaks above the noise level."
        )
        filter_grid.addWidget(precheck_kmad_label, r, 0)
        filter_grid.addWidget(self.precheck_kmad_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # DBSCAN params (spatial outlier filtering)
        self.dbscan_enabled_check = QtWidgets.QCheckBox("DBSCAN Outlier Removal")
        self.dbscan_enabled_check.setChecked(bool(getattr(self.spot_analyzer, "dbscan_enabled", False)))
        self.dbscan_enabled_check.setToolTip("Removes spatially isolated peaks as noise.")
        filter_grid.addWidget(self.dbscan_enabled_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.dbscan_eps_spin = QtWidgets.QDoubleSpinBox()
        self.dbscan_eps_spin.setRange(0.5, 50.0)
        self.dbscan_eps_spin.setSingleStep(0.5)
        self.dbscan_eps_spin.setValue(float(getattr(self.spot_analyzer, "dbscan_eps", 5.0)))
        self.dbscan_eps_spin.setMaximumWidth(90)
        self.dbscan_eps_spin.setToolTip("DBSCAN neighborhood radius in pixels.")
        dbscan_eps_label = QtWidgets.QLabel("DBSCAN eps")
        dbscan_eps_label.setToolTip(
            "Neighborhood radius for DBSCAN clustering in pixels:\n"
            "Used by the Multiscale (LoG+DBSCAN) initialization mode.\n"
            "Points within this distance are treated as the same cluster to remove duplicate detections.\n"
            "A value near the minimum peak distance is often a good starting point."
        )
        filter_grid.addWidget(dbscan_eps_label, r, 0)
        filter_grid.addWidget(self.dbscan_eps_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.min_amp_spin = QtWidgets.QDoubleSpinBox()
        self.min_amp_spin.setRange(0.0, 1000.0)
        self.min_amp_spin.setSingleStep(0.1)
        self.min_amp_spin.setValue(self.spot_analyzer.min_amplitude)
        self.min_amp_spin.setMaximumWidth(90)
        min_amp_label = QtWidgets.QLabel("Minimum Amplitude")
        min_amp_label.setToolTip(
            "Minimum amplitude for peaks included in final results:\n"
            "Peaks whose fitted Gaussian amplitude (height) is below this value\n"
            "are excluded from final results."
        )
        filter_grid.addWidget(min_amp_label, r, 0)
        filter_grid.addWidget(self.min_amp_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.min_sigma_result_spin = QtWidgets.QDoubleSpinBox()
        self.min_sigma_result_spin.setRange(0.0, 50.0)
        self.min_sigma_result_spin.setSingleStep(0.1)
        self.min_sigma_result_spin.setValue(self.spot_analyzer.min_sigma_result)
        self.min_sigma_result_spin.setMaximumWidth(90)
        min_sigma_result_label = QtWidgets.QLabel("Minimum Sigma (result)")
        min_sigma_result_label.setToolTip(
            "Minimum sigma for peaks included in final results:\n"
            "Peaks whose fitted sigma is below this value are excluded.\n"
            "Use this to remove extremely small point-like peaks as noise."
        )
        filter_grid.addWidget(min_sigma_result_label, r, 0)
        filter_grid.addWidget(self.min_sigma_result_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(filter_grid, r)

        form_layout.addWidget(filter_group)

        view_group = QtWidgets.QGroupBox("Display / Recording")
        view_grid = QtWidgets.QGridLayout(view_group)
        _setup_grid(view_grid)
        r = 0
        self.spot_radius_spin = QtWidgets.QSpinBox()
        self.spot_radius_spin.setRange(1, 200)
        self.spot_radius_spin.setValue(4)
        self.spot_radius_spin.setToolTip("Records the average height inside a circle of radius r(px) around each spot center.")
        self.spot_radius_spin.setMaximumWidth(80)
        self.spot_radius_spin.valueChanged.connect(self._on_spot_radius_changed)
        spot_radius_label = QtWidgets.QLabel("Spot Radius (px)")
        spot_radius_label.setToolTip(
            "Spot measurement radius in pixels:\n"
            "Calculates and records the average height inside a circle of radius r(px) from the detected peak center.\n"
            "This value is also used for display as the peak marker circle size."
        )
        view_grid.addWidget(spot_radius_label, r, 0)
        view_grid.addWidget(self.spot_radius_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.show_roi_spots_check = QtWidgets.QCheckBox("Show Spots on ROI Images")
        self.show_roi_spots_check.setChecked(True)
        self.show_roi_spots_check.setToolTip("Overlay spots on the lower ROI images (preprocessed ROI / LoG or DoG ROI).")
        self.show_roi_spots_check.toggled.connect(self._on_any_ui_changed)
        view_grid.addWidget(self.show_roi_spots_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.show_det_spots_check = QtWidgets.QCheckBox("Show Spots on Detection Image")
        self.show_det_spots_check.setChecked(True)
        self.show_det_spots_check.setToolTip("Overlay spots on the lower detection image (LoG/DoG).")
        self.show_det_spots_check.toggled.connect(self._on_any_ui_changed)
        view_grid.addWidget(self.show_det_spots_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Debug: show fit-result (final) peaks on ROI(preprocessed)
        self.show_fit_spots_on_roi_pre_check = QtWidgets.QCheckBox("Show Fitted Peaks on ROI(pre) (debug)")
        self.show_fit_spots_on_roi_pre_check.setChecked(False)
        self.show_fit_spots_on_roi_pre_check.setToolTip(
            "Overlay final fitted peak positions on the lower-left ROI(preprocessed) image.\n"
            "Use this to check consistency between LoG/DoG initial positions and fitted results."
        )
        self.show_fit_spots_on_roi_pre_check.toggled.connect(self._on_any_ui_changed)
        view_grid.addWidget(self.show_fit_spots_on_roi_pre_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Snap-to-local-max (final peak coordinates)
        self.snap_enabled_check = QtWidgets.QCheckBox("Snap Final Peaks to Local Maxima")
        self.snap_enabled_check.setChecked(bool(getattr(self.spot_analyzer, "snap_enabled", False)))
        self.snap_enabled_check.setToolTip("Move final peak coordinates to local maxima on the detection image (LoG/DoG).")
        view_grid.addWidget(self.snap_enabled_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.snap_radius_spin = QtWidgets.QSpinBox()
        self.snap_radius_spin.setRange(0, 50)
        self.snap_radius_spin.setValue(int(getattr(self.spot_analyzer, "snap_radius", 2)))
        self.snap_radius_spin.setMaximumWidth(80)
        self.snap_radius_spin.setToolTip("Snap search radius (px). 0 disables snapping.")
        snap_radius_label = QtWidgets.QLabel("Snap Radius (px)")
        snap_radius_label.setToolTip(
            "Snap search radius in pixels:\n"
            "When moving final peaks to local maxima on the detection image (LoG/DoG),\n"
            "the maximum is searched inside this radius.\n"
            "Set to 0 to disable snapping."
        )
        view_grid.addWidget(snap_radius_label, r, 0)
        view_grid.addWidget(self.snap_radius_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.snap_refit_check = QtWidgets.QCheckBox("Refit Once After Snap")
        self.snap_refit_check.setChecked(bool(getattr(self.spot_analyzer, "snap_refit_enabled", False)))
        self.snap_refit_check.setToolTip("Use snapped positions as initial values and refit once with the same number of peaks. This may be slower.")
        view_grid.addWidget(self.snap_refit_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        _add_right_spacer(view_grid, r)

        form_layout.addWidget(view_group)

        # --- Status / output ---
        self.roi_status_label = QtWidgets.QLabel("ROI Not Selected")
        form_layout.addWidget(self.roi_status_label)

        self.selection_label = QtWidgets.QLabel("")
        form_layout.addWidget(self.selection_label)

        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        try:
            fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self.output.setFont(fixed_font)
        except Exception:
            pass
        try:
            self.output.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        except Exception:
            pass
        form_layout.addWidget(self.output)

        height_export_row = QtWidgets.QHBoxLayout()
        height_export_label = QtWidgets.QLabel("Height Value for CSV")
        height_export_label.setToolTip(
            "Select the primary height value saved to CSV.\n"
            "- Spot Position: height at the spot coordinate\n"
            "- Spot-Radius Mean: average height inside the spot-radius circle"
        )
        height_export_row.addWidget(height_export_label)
        self.height_export_mode_combo = QtWidgets.QComboBox()
        self.height_export_mode_combo.addItems(["Spot Position", "Spot-Radius Mean"])
        self.height_export_mode_combo.setCurrentText("Spot-Radius Mean")
        self.height_export_mode_combo.setToolTip("Select which height value is saved as the primary CSV height.")
        height_export_row.addWidget(self.height_export_mode_combo)
        height_export_row.addStretch(1)
        form_layout.addLayout(height_export_row)

        self.export_btn = QtWidgets.QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_spots_csv)
        self.export_btn.setEnabled(False)
        form_layout.addWidget(self.export_btn)

        self.import_resume_btn = QtWidgets.QPushButton("Load CSV -> Restore")
        self.import_resume_btn.setToolTip("Load saved Spot CSV files and restore the in-progress spot display state.")
        self.import_resume_btn.clicked.connect(self.import_csv_restore)
        form_layout.addWidget(self.import_resume_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset Analysis Results")
        self.reset_btn.setToolTip("Clear spot results, ROI, and display state together manually.")
        self.reset_btn.clicked.connect(self._reset_analysis_results)
        form_layout.addWidget(self.reset_btn)

        self.edit_help_label = QtWidgets.QLabel(
            "Spot editing: Ctrl/Cmd+drag=move, Shift+click=add, Alt(Option)+click=delete"
        )
        self.edit_help_label.setStyleSheet("color: #666; font-size: 11px;")
        form_layout.addWidget(self.edit_help_label)

        # --- Add form_widget to content layout (left-aligned) ---
        layout.addWidget(form_widget, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch(1)

        # --- Live updates ---
        # Any UI changes should redraw immediately; analysis-impacting widgets also trigger debounced reanalysis.
        for w in (
            # analysis-impacting controls
            self.median_check,
            self.open_check,
            self.detection_mode_combo,
            self.bandpass_low_spin,
            self.bandpass_high_spin,
            self.log_sigma_spin,
            self.median_size_spin,
            self.open_radius_spin,
            self.margin_spin,
            self.min_amp_spin,
            self.min_sigma_result_spin,
            self.criterion_combo,
            self.min_peaks_spin,
            self.max_peaks_spin,
            self.init_mode_combo,
            self.subpixel_check,
            self.watershed_h_rel_spin,
            self.watershed_adaptive_h_check,
            self.snr_spin,
            self.peak_min_distance_spin,
            self.localmax_thr_spin,
            self.localmax_snr_spin,
            self.precheck_radius_spin,
            self.precheck_kmad_spin,
            self.blob_doh_min_sigma_spin,
            self.blob_doh_max_sigma_spin,
            self.blob_doh_threshold_rel_spin,
            self.dbscan_enabled_check,
            self.dbscan_eps_spin,
            self.fit_enabled_check,
            self.auto_centroid_check,
            self.initial_sigma_spin,
            self.sigma_min_spin,
            self.sigma_max_spin,
            self.snap_enabled_check,
            self.snap_radius_spin,
            self.snap_refit_check,
        ):
            try:
                if isinstance(w, QtWidgets.QAbstractButton):
                    w.toggled.connect(self._on_analysis_param_changed)
                elif hasattr(w, "valueChanged"):
                    w.valueChanged.connect(self._on_analysis_param_changed)
                elif hasattr(w, "currentTextChanged"):
                    w.currentTextChanged.connect(self._on_analysis_param_changed)
            except Exception:
                pass

        self._update_detection_ui_enabled()
        self._update_init_mode_ui_enabled()
        self._update_fit_ui_enabled()
        self._sync_run_buttons_enabled()

    def _update_detection_ui_enabled(self, *_args) -> None:
        mode = (self.detection_mode_combo.currentText() or "DoG").strip().lower()
        is_log = mode == "log"
        is_pre = mode in ("pre", "preprocessed")
        # DoG widgets
        for w in (self.bandpass_low_spin, self.bandpass_high_spin):
            w.setEnabled(not is_log and not is_pre)
        # LoG widget
        self.log_sigma_spin.setEnabled(is_log and not is_pre)
        # NOTE: redraw/reanalysis is handled by the parameter-change handler

    def _update_init_mode_ui_enabled(self, *_args) -> None:
        mode = (self.init_mode_combo.currentText() or "Watershed (Recommended)").strip().lower()
        is_doh = "doh" in mode
        is_ws = "watershed" in mode or "water" in mode
        is_peak = "peak" in mode

        # DoH parameters
        for w in (
            getattr(self, "blob_doh_min_sigma_spin", None),
            getattr(self, "blob_doh_max_sigma_spin", None),
            getattr(self, "blob_doh_threshold_rel_spin", None),
        ):
            try:
                if w is not None:
                    w.setEnabled(bool(is_doh))
            except Exception:
                pass

        # Watershed parameters
        for w in (
            getattr(self, "watershed_h_rel_spin", None),
            getattr(self, "watershed_adaptive_h_check", None),
        ):
            try:
                if w is not None:
                    w.setEnabled(bool(is_ws))
            except Exception:
                pass

        # Peak parameters
        try:
            if getattr(self, "localmax_thr_spin", None) is not None:
                self.localmax_thr_spin.setEnabled(bool(is_peak))
            if getattr(self, "localmax_snr_spin", None) is not None:
                self.localmax_snr_spin.setEnabled(bool(is_peak))
            if getattr(self, "subpixel_check", None) is not None:
                self.subpixel_check.setEnabled(bool(is_peak))
        except Exception:
            pass

    def _update_fit_ui_enabled(self, *_args) -> None:
        fit_enabled = bool(getattr(self, "fit_enabled_check", None) and self.fit_enabled_check.isChecked())
        try:
            if getattr(self, "snap_enabled_check", None) is not None:
                self.snap_enabled_check.setEnabled(fit_enabled)
                if not fit_enabled:
                    self.snap_enabled_check.setToolTip(
                        "When fitting is OFF, snapping is disabled because coordinates from initial-position search are used directly."
                    )
                else:
                    self.snap_enabled_check.setToolTip(
                        "Move final peak coordinates to local maxima on the detection image (LoG/DoG)."
                    )
            if getattr(self, "snap_radius_spin", None) is not None:
                self.snap_radius_spin.setEnabled(fit_enabled)
            if getattr(self, "snap_refit_check", None) is not None:
                self.snap_refit_check.setEnabled(fit_enabled)
                if not fit_enabled:
                    self.snap_refit_check.setToolTip(
                        "Refitting is unavailable while Gaussian fitting is OFF."
                    )
                else:
                    self.snap_refit_check.setToolTip(
                        "Use snapped positions as initial values and refit once with the same number of peaks. This may be slower."
                    )
        except Exception:
            pass

    def _sync_run_buttons_enabled(self) -> None:
        has_roi = self.manual_roi is not None
        try:
            self.run_btn.setEnabled(bool(has_roi))
        except Exception:
            pass
        try:
            allow_all = bool(has_roi) and bool(self.run_all_enable_check.isChecked())
            self.run_all_btn.setEnabled(allow_all)
        except Exception:
            pass
        try:
            # manual refit requires existing spots on the current frame
            fi = self._get_current_frame_index()
            has_spots = bool(self.spots_by_frame.get(fi))
            self.refit_manual_btn.setEnabled(bool(has_roi) and has_spots)
        except Exception:
            try:
                self.refit_manual_btn.setEnabled(False)
            except Exception:
                pass

    def _on_any_ui_changed(self, *_args) -> None:
        """Immediate UI-change update for display only."""
        self._refresh_overlay()

    def _on_analysis_param_changed(self, *_args) -> None:
        """
        When analysis parameters change:
        - Display elements such as ROI images update immediately
        - Analysis updates for spots are debounced
        """
        self._refresh_overlay()
        self._schedule_reanalysis()

    # Backward-compat: keep old handler name used by some connections.
    def _on_params_changed(self, *_args) -> None:
        self._on_analysis_param_changed(*_args)

    def _schedule_reanalysis(self) -> None:
        if self.manual_roi is None:
            return
        try:
            # Debounce spinbox changes so analysis is not triggered repeatedly.
            self._reanalysis_timer.start(250)
        except Exception:
            pass

    def _should_skip_auto_analysis_for_frame(self, frame_index: int) -> bool:
        """
        When CSV-restored spots are being reviewed, keep existing per-frame spots
        unless the user explicitly runs analysis.
        """
        if not bool(getattr(self, "preserve_existing_spots_on_auto", False)):
            return False
        try:
            return bool(self.spots_by_frame.get(int(frame_index)))
        except Exception:
            return False

    def _analysis_signature(self, frame_index: int, roi_info: Optional[Dict[str, float]]) -> Tuple:
        def _rf(v: float) -> float:
            try:
                return round(float(v), 6)
            except Exception:
                return float("nan")

        if roi_info is None:
            roi_sig: Tuple = ("none",)
        else:
            shape = str(roi_info.get("shape", "Rectangle"))
            roi_sig = (
                shape,
                _rf(roi_info.get("x0", 0.0)),
                _rf(roi_info.get("y0", 0.0)),
                _rf(roi_info.get("w", 0.0)),
                _rf(roi_info.get("h", 0.0)),
                _rf(roi_info.get("cx", 0.0)) if shape == "Ellipse" else None,
                _rf(roi_info.get("cy", 0.0)) if shape == "Ellipse" else None,
                _rf(roi_info.get("rx", 0.0)) if shape == "Ellipse" else None,
                _rf(roi_info.get("ry", 0.0)) if shape == "Ellipse" else None,
            )

        # UI -> params signature
        crit = (self.criterion_combo.currentText() or "AIC").strip().lower()
        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())

        mode_ui = (self.detection_mode_combo.currentText() or "DoG").strip().lower()
        if mode_ui in ("pre", "preprocessed"):
            det_mode = "pre"
            det_sig = ("pre",)
        elif mode_ui == "log":
            det_mode = "log"
            det_sig = ("log", _rf(self.log_sigma_spin.value()))
        else:
            det_mode = "dog"
            det_sig = ("dog", _rf(self.bandpass_low_spin.value()), _rf(self.bandpass_high_spin.value()))

        params_sig = (
            crit,
            min_peaks,
            max_peaks,
            det_sig,
            (self.init_mode_combo.currentText() or "blob_log").strip().lower(),
            bool(getattr(self, "subpixel_check", None) is not None and self.subpixel_check.isChecked()),
            _rf(self.watershed_h_rel_spin.value()),
            bool(self.watershed_adaptive_h_check.isChecked()),
            (_rf(self.blob_doh_min_sigma_spin.value()), _rf(self.blob_doh_max_sigma_spin.value())),
            _rf(self.blob_doh_threshold_rel_spin.value()),
            bool(self.dbscan_enabled_check.isChecked()),
            _rf(self.dbscan_eps_spin.value()),
            bool(self.median_check.isChecked()),
            int(self.median_size_spin.value()),
            bool(self.open_check.isChecked()),
            int(self.open_radius_spin.value()),
            int(self.peak_min_distance_spin.value()),
            _rf(self.localmax_thr_spin.value()),
            _rf(self.localmax_snr_spin.value()),
            int(self.precheck_radius_spin.value()),
            _rf(self.precheck_kmad_spin.value()),
            bool(self.fit_enabled_check.isChecked()),
            bool(self.auto_centroid_check.isChecked()),
            _rf(self.initial_sigma_spin.value()),
            (_rf(self.sigma_min_spin.value()), _rf(self.sigma_max_spin.value())),
            _rf(self.snr_spin.value()),
            # results filters
            int(self.margin_spin.value()),
            _rf(self.min_amp_spin.value()),
            _rf(self.min_sigma_result_spin.value()),
            # final peak snapping
            bool(self.snap_enabled_check.isChecked()),
            int(self.snap_radius_spin.value()),
            bool(self.snap_refit_check.isChecked()),
        )
        return (int(frame_index), roi_sig, params_sig)

    def _reanalyze_current_frame_debounced(self) -> None:
        if self._auto_busy:
            return
        if self.manual_roi is None:
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return
        if not self._apply_ui_to_analyzer(show_errors=False):
            return

        frame_index = self._get_current_frame_index()
        if self._should_skip_auto_analysis_for_frame(frame_index):
            return
        roi_info = self._current_roi_overlay()
        sig = self._analysis_signature(frame_index, roi_info)
        if self._analysis_signature_by_frame.get(frame_index) == sig:
            return

        criterion = self.criterion_combo.currentText().lower()
        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            return

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
        self._auto_busy = True
        try:
            self._analyze_current_frame(
                frame,
                criterion,
                center_override,
                roi_size_override,
                roi_mask,
                roi_bounds,
                min_peaks,
                max_peaks,
                show_errors=False,
            )
        finally:
            self._auto_busy = False

    def _apply_ui_to_analyzer(self, show_errors: bool) -> bool:
        # preprocess
        self.spot_analyzer.median_enabled = bool(self.median_check.isChecked())
        self.spot_analyzer.median_size = int(self.median_size_spin.value())
        self.spot_analyzer.open_enabled = bool(self.open_check.isChecked())
        self.spot_analyzer.open_radius = int(self.open_radius_spin.value())

        # detection mode
        mode = (self.detection_mode_combo.currentText() or "DoG").strip().lower()
        if mode in ("pre", "preprocessed"):
            self.spot_analyzer.detection_mode = "pre"
        elif mode == "log":
            self.spot_analyzer.detection_mode = "log"
        else:
            self.spot_analyzer.detection_mode = "dog"

        if self.spot_analyzer.detection_mode == "dog":
            low_sigma = float(self.bandpass_low_spin.value())
            high_sigma = float(self.bandpass_high_spin.value())
            if high_sigma <= low_sigma:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "DoG: sigma high must be greater than sigma low.")
                return False
            self.spot_analyzer.bandpass_low_sigma = low_sigma
            self.spot_analyzer.bandpass_high_sigma = high_sigma
        elif self.spot_analyzer.detection_mode == "log":
            log_sigma = float(self.log_sigma_spin.value())
            if log_sigma <= 0:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "LoG: sigma must be positive.")
                return False
            self.spot_analyzer.log_sigma = log_sigma

        # fit params
        init_ui = (self.init_mode_combo.currentText() or "Watershed (Recommended)").strip().lower()
        if "watershed" in init_ui or "water" in init_ui:
            self.spot_analyzer.init_mode = "watershed"
        elif "doh" in init_ui or "blob doh" in init_ui:
            self.spot_analyzer.init_mode = "blob_doh"
        elif "peak" in init_ui:
            # Check subpixel option
            if hasattr(self, "subpixel_check") and self.subpixel_check.isChecked():
                self.spot_analyzer.init_mode = "peak_subpixel"
            else:
                self.spot_analyzer.init_mode = "peak"
        else:
            # Default fallback
            self.spot_analyzer.init_mode = "watershed"
        self.spot_analyzer.watershed_h_rel = float(self.watershed_h_rel_spin.value())
        self.spot_analyzer.watershed_adaptive_h = bool(self.watershed_adaptive_h_check.isChecked())
        self.spot_analyzer.blob_doh_min_sigma = float(self.blob_doh_min_sigma_spin.value())
        self.spot_analyzer.blob_doh_max_sigma = float(self.blob_doh_max_sigma_spin.value())
        self.spot_analyzer.blob_doh_num_sigma = int(getattr(self.spot_analyzer, "blob_doh_num_sigma", 10))
        self.spot_analyzer.blob_doh_threshold_rel = float(self.blob_doh_threshold_rel_spin.value())
        self.spot_analyzer.dbscan_enabled = bool(self.dbscan_enabled_check.isChecked())
        self.spot_analyzer.dbscan_eps = float(self.dbscan_eps_spin.value())

        self.spot_analyzer.peak_min_distance = max(1, int(self.peak_min_distance_spin.value()))
        self.spot_analyzer.localmax_threshold_rel = float(self.localmax_thr_spin.value())
        self.spot_analyzer.localmax_threshold_snr = float(self.localmax_snr_spin.value())
        try:
            self.spot_analyzer.precheck_radius_px = max(0, int(self.precheck_radius_spin.value()))
        except Exception:
            self.spot_analyzer.precheck_radius_px = max(0, int(getattr(self.spot_analyzer, "precheck_radius_px", 2)))
        try:
            self.spot_analyzer.precheck_kmad = float(self.precheck_kmad_spin.value())
        except Exception:
            self.spot_analyzer.precheck_kmad = float(getattr(self.spot_analyzer, "precheck_kmad", 1.0))
        self.spot_analyzer.fit_enabled = bool(self.fit_enabled_check.isChecked())
        self.spot_analyzer.initial_sigma = float(self.initial_sigma_spin.value())
        sigma_min = float(self.sigma_min_spin.value())
        sigma_max = float(self.sigma_max_spin.value())
        if sigma_max <= sigma_min:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "The sigma upper bound must be greater than the sigma lower bound.")
            return False
        self.spot_analyzer.sigma_bounds = (sigma_min, sigma_max)
        self.spot_analyzer.snr_threshold = float(self.snr_spin.value())

        # filters
        self.spot_analyzer.margin = int(self.margin_spin.value())
        self.spot_analyzer.min_amplitude = float(self.min_amp_spin.value())
        self.spot_analyzer.min_sigma_result = float(self.min_sigma_result_spin.value())

        # final peak snapping (display/output coordinates)
        self.spot_analyzer.snap_enabled = bool(self.snap_enabled_check.isChecked())
        self.spot_analyzer.snap_radius = max(0, int(self.snap_radius_spin.value()))
        self.spot_analyzer.snap_refit_enabled = bool(self.snap_refit_check.isChecked())

        return True

    def _refresh_selection_label(self) -> None:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            self.selection_label.setText("Could not get file-selection information.")
            return
        selected = self.main_window.FileList.selectedItems()
        if selected:
            names = [item.text() for item in selected]
            self.selection_label.setText(f"Selected: {', '.join(names)}")
        else:
            current = self.main_window.FileList.currentItem()
            if current:
                self.selection_label.setText(f"Current: {current.text()}")
            else:
                self.selection_label.setText("No selection")

    def _ensure_selection_loaded(self) -> bool:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            QtWidgets.QMessageBox.warning(self, "No Selection", "FileList was not found.")
            return False
        selected = self.main_window.FileList.selectedIndexes()
        if selected:
            target_row = selected[0].row()
        else:
            target_row = self.main_window.FileList.currentRow()
        if target_row is None or target_row < 0:
            QtWidgets.QMessageBox.information(self, "Select File", "Select a target in the file list.")
            return False

        # Reflect selection according to MainWindow logic
        # setCurrentRow emits itemSelectionChanged, and the normal pyNuD handler may
        # bring the main window to front, so block signals temporarily to avoid that.
        try:
            blocker = QtCore.QSignalBlocker(self.main_window.FileList)
            self.main_window.FileList.setCurrentRow(target_row)
        except Exception:
            self.main_window.FileList.setCurrentRow(target_row)
        if hasattr(self.main_window, "ListClickFunction"):
            try:
                # Do not bring the main window to front when called from SpotAnalysis.
                try:
                    self.main_window.ListClickFunction(bring_to_front=False)
                except TypeError:
                    # Compatibility with the old signature
                    self.main_window.ListClickFunction()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load file:\n{exc}")
                return False
        return True

    def _prepare_frame(self) -> Optional[np.ndarray]:
        if not hasattr(gv, "files") or not gv.files:
            QtWidgets.QMessageBox.information(self, "No Files", "No files are loaded.")
            return None
        if getattr(gv, "currentFileNum", -1) < 0 or gv.currentFileNum >= len(gv.files):
            QtWidgets.QMessageBox.warning(self, "Invalid Selection", "The selected file index is invalid.")
            return None

        # Get the latest frame
        try:
            LoadFrame(gv.files[gv.currentFileNum])
            InitializeAryDataFallback()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load frame:\n{exc}")
            return None

        if not hasattr(gv, "aryData") or gv.aryData is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Image data is not available.")
            return None
        frame = np.asarray(gv.aryData, dtype=np.float64)
        if frame.ndim != 2:
            QtWidgets.QMessageBox.warning(self, "Data Error", "Only 2D image data can be analyzed.")
            return None
        return frame

    def _connect_frame_signal(self) -> None:
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_frame_changed)
            except Exception:
                pass

    def _on_frame_changed(self, frame_index: int) -> None:
        # Update the current frame display
        if self.auto_analyze_check.isChecked():
            # During Auto Analysis:
            #  - Prefer the manual ROI for that frame if available
            #  - Otherwise inherit the most recent ROI from a previous frame (propagation)
            roi_here = self.roi_by_frame.get(frame_index)
            if roi_here is not None:
                self.manual_roi = roi_here
            else:
                prev_roi = self._get_last_roi_at_or_before(frame_index - 1)
                if prev_roi is not None:
                    # Save a copy to avoid unintended mutation through shared references
                    propagated = dict(prev_roi)
                    self.roi_by_frame[frame_index] = propagated
                    self.manual_roi = propagated
                else:
                    self.manual_roi = None
        else:
            self.manual_roi = self.roi_by_frame.get(frame_index)
        if self.manual_roi is None:
            self.roi_status_label.setText("ROI Not Selected")
        else:
            self.roi_status_label.setText("ROI Selected")
        self._sync_run_buttons_enabled()
        frame = self._prepare_frame()
        if frame is None:
            return
        self.last_frame = frame
        # Always pass both fit-spots and init-spots; rendering decides where to show them.
        spots = self.spots_by_frame.get(frame_index)
        initial_spots = self.initial_spots_by_frame.get(frame_index)
        roi_overlay = self._current_roi_overlay()
        if self.full_viz_window and self._is_window_live(self.full_viz_window):
            self._apply_ui_to_analyzer(show_errors=False)
            _co, _rs, roi_mask, roi_bounds = self._roi_overrides(self.last_frame.shape)
            roi_pre, roi_det = self.spot_analyzer.compute_roi_visual_images(
                self.last_frame, roi_mask_override=roi_mask, roi_bounds_override=roi_bounds
            )
            # Use last_result only when it likely corresponds to the current frame.
            disp_result = self.last_result if frame_index == self._get_current_frame_index() else None
            self.full_viz_window.update_view(
                self.last_frame,
                disp_result,
                spots,
                initial_spots,
                roi_overlay,
                self._get_spot_radius_px(),
                roi_pre_image=roi_pre,
                roi_det_image=roi_det,
                det_label=self.spot_analyzer.detection_label(),
                show_det_spots=bool(getattr(self, "show_det_spots_check", None) and self.show_det_spots_check.isChecked()),
                show_initial_spots=False,
                show_fit_spots_on_roi_pre=bool(
                    getattr(self, "show_fit_spots_on_roi_pre_check", None)
                    and self.show_fit_spots_on_roi_pre_check.isChecked()
                ),
                init_mode=str(getattr(self.spot_analyzer, "init_mode", "") or ""),
            )
        self._update_frame_label()
        if self.auto_analyze_check.isChecked():
            self._auto_analyze_if_enabled(frame_index, frame)

    def _get_current_frame_index(self) -> int:
        return int(getattr(gv, "index", 0))

    def _get_last_roi_at_or_before(self, frame_index: int) -> Optional[Dict[str, float]]:
        """
        Return the most recently set ROI at or before the specified frame_index.
        Returns None when no ROI exists.
        """
        if frame_index < 0:
            return None
        for idx in range(int(frame_index), -1, -1):
            roi = self.roi_by_frame.get(idx)
            if roi is not None:
                return roi
        return None

    def _update_frame_label(self) -> None:
        total = int(getattr(gv, "FrameNum", 0))
        current = self._get_current_frame_index()
        if total > 0:
            self.frame_label.setText(f"Frame: {current + 1} / {total}")
        else:
            self.frame_label.setText("Frame: - / -")

    def _prev_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0))
        if total <= 0:
            return
        new_index = max(0, self._get_current_frame_index() - 1)
        self._set_frame_index(new_index)

    def _next_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0))
        if total <= 0:
            return
        new_index = min(total - 1, self._get_current_frame_index() + 1)
        self._set_frame_index(new_index)

    def _set_frame_index(self, index: int) -> None:
        gv.index = int(index)
        if self.main_window and hasattr(self.main_window, "frameSlider"):
            try:
                self.main_window.frameSlider.setValue(gv.index)
            except Exception:
                pass
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.emit(gv.index)
            except Exception:
                self._on_frame_changed(gv.index)

    def run_analysis(self) -> None:
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "Select an ROI first.")
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return

        if not self._apply_ui_to_analyzer(show_errors=True):
            return
        criterion = self.criterion_combo.currentText().lower()

        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "The maximum peak count must be greater than or equal to the minimum peak count.")
            return

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)

        self._analyze_current_frame(
            frame,
            criterion,
            center_override,
            roi_size_override,
            roi_mask,
            roi_bounds,
            min_peaks,
            max_peaks,
            show_errors=True,
        )

    def refit_from_manual_spots(self) -> None:
        """
        Move current manual spots to the intensity centroid inside the
        Spot-radius circle drawn for each spot.
        """
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "Select an ROI first.")
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return
        if not self._apply_ui_to_analyzer(show_errors=True):
            return

        frame_index = self._get_current_frame_index()
        if not self._apply_spot_centroid_to_frame(
            frame_index,
            frame=frame,
            result=self.last_result,
            show_messages=True,
            refresh_display=True,
        ):
            return

        self._refresh_overlay()
        self.export_btn.setEnabled(True)

        # Update signature so debounce reanalysis won't immediately overwrite
        try:
            self._analysis_signature_by_frame[frame_index] = self._analysis_signature(frame_index, self._current_roi_overlay())
        except Exception:
            pass

    def _apply_spot_centroid_to_frame(
        self,
        frame_index: int,
        frame: Optional[np.ndarray] = None,
        result: Optional[FrameAnalysis] = None,
        show_messages: bool = False,
        refresh_display: bool = False,
    ) -> bool:
        spots = list(self.spots_by_frame.get(frame_index) or [])
        if not spots:
            if show_messages:
                QtWidgets.QMessageBox.information(self, "No Spots", "No manual spots are available. Add or move spots before refitting.")
            return False

        use_frame = frame if frame is not None else self.last_frame
        if use_frame is None:
            if show_messages:
                QtWidgets.QMessageBox.warning(self, "No Data", "Image data is not available.")
            return False

        reference_spots = list(self.centroid_reference_by_frame.get(frame_index) or [])
        if len(reference_spots) != len(spots):
            reference_spots = [dict(pk) for pk in spots]
            self.centroid_reference_by_frame[frame_index] = [dict(pk) for pk in reference_spots]

        seeds_abs: List[Tuple[float, float]] = []
        for pk in reference_spots:
            try:
                x = float(pk.get("x", float("nan")))
                y = float(pk.get("y", float("nan")))
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            seeds_abs.append((x, y))
        if not seeds_abs:
            if show_messages:
                QtWidgets.QMessageBox.warning(self, "Invalid Spots", "Manual spot coordinates are invalid.")
            return False

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(use_frame.shape)
        frame_f = np.asarray(use_frame, dtype=np.float64)
        if roi_bounds is not None:
            roi_raw, origin = self.spot_analyzer._crop_rect(frame_f, roi_bounds)
        else:
            center = center_override if center_override is not None else (frame_f.shape[1] / 2.0, frame_f.shape[0] / 2.0)
            use_roi_size = roi_size_override if roi_size_override is not None else self.spot_analyzer.roi_size
            roi_raw, origin = self.spot_analyzer._crop_square(frame_f, center, use_roi_size)
        h_roi, w_roi = roi_raw.shape
        centroid_radius = max(1, int(self._get_spot_radius_px()))

        initial_local: List[Tuple[float, float]] = []
        roi_mask_bool = None
        if roi_mask is not None:
            try:
                roi_mask_bool = roi_mask.astype(bool)
            except Exception:
                roi_mask_bool = None

        for x_abs, y_abs in seeds_abs:
            lx = float(x_abs) - float(origin[0])
            ly = float(y_abs) - float(origin[1])
            if not (0.0 <= lx < float(w_roi) and 0.0 <= ly < float(h_roi)):
                if show_messages:
                    QtWidgets.QMessageBox.warning(self, "Spot Outside ROI", "Manual spots are outside the ROI. Move them inside the ROI before refitting.")
                return False
            if roi_mask_bool is not None:
                try:
                    ix = int(round(lx))
                    iy = int(round(ly))
                    if ix < 0 or iy < 0 or ix >= w_roi or iy >= h_roi or not bool(roi_mask_bool[iy, ix]):
                        if show_messages:
                            QtWidgets.QMessageBox.warning(self, "Spot Outside ROI Mask", "Manual spots are outside the ROI mask, such as an ellipse. Move them inside the mask before refitting.")
                        return False
                except Exception:
                    pass
            initial_local.append((lx, ly))

        if not initial_local:
            if show_messages:
                QtWidgets.QMessageBox.warning(self, "No Valid Spots", "No valid spots are available for refitting.")
            return False

        self.last_frame = use_frame
        moved_spots: List[Dict[str, float]] = []
        for src, (lx, ly) in zip(spots, initial_local):
            centroid = self._compute_circle_centroid_in_roi(
                roi_raw,
                float(lx),
                float(ly),
                radius_px=float(centroid_radius),
                roi_mask=roi_mask,
            )
            if centroid is None:
                cx, cy = float(lx), float(ly)
            else:
                cx, cy = centroid
            moved = dict(src)
            moved["x"] = float(origin[0] + cx)
            moved["y"] = float(origin[1] + cy)
            moved_spots.append(moved)

        self.spots_by_frame[frame_index] = moved_spots
        self._recompute_spot_heights_for_frame(frame_index, frame=use_frame)

        target_result = result
        if target_result is None and self.last_result is not None:
            target_result = self.last_result
        if target_result is not None and target_result.best_n_peaks in target_result.models:
            try:
                best_model = target_result.models[target_result.best_n_peaks]
                if len(best_model.peaks) == len(moved_spots):
                    for pk, moved in zip(best_model.peaks, moved_spots):
                        pk.x = float(moved["x"])
                        pk.y = float(moved["y"])
                    if refresh_display:
                        self._display_result(target_result)
            except Exception:
                pass

        return True

    def _display_result(self, result: FrameAnalysis) -> None:
        def esc(s: str) -> str:
            return html.escape(s, quote=False)

        html_lines: List[str] = []
        best_model = result.models.get(result.best_n_peaks) if result.best_n_peaks in result.models else None
        fit_suffix = ""
        if best_model is not None and not bool(getattr(best_model, "fit_applied", True)):
            fit_suffix = ", No Fit"
        html_lines.append(esc(f"Selected model: {result.best_n_peaks} peaks ({result.criterion.upper()}{fit_suffix})"))
        for n_peaks in sorted(result.models.keys()):
            model = result.models[n_peaks]
            mode_label = "Fit" if bool(getattr(model, "fit_applied", True)) else "NoFit"
            html_lines.append(
                esc(
                    f"[{n_peaks} peaks, {mode_label}] AIC={model.aic:.2f}, BIC={model.bic:.2f}, rss={model.rss:.4g}, loglike={model.loglike:.4g}, residual_std={model.residual_std:.4g}"
                )
            )
            for idx, pk in enumerate(model.peaks, start=1):
                html_lines.append(
                    esc(
                        f"  P{idx}: amp={pk.amplitude:.3g}, sigma={pk.sigma:.3g}, (x,y)=({pk.x:.2f},{pk.y:.2f}), S/N={pk.snr:.2f}"
                    )
                )

        # Show excluded-peak information for the best model only
        if result.best_n_peaks is not None and result.best_n_peaks in result.models:
            best_model = result.models[result.best_n_peaks]
            init_peaks = best_model.init_peaks if hasattr(best_model, 'init_peaks') else []
            final_peaks = best_model.peaks
            excluded_infos = list(getattr(best_model, "excluded_infos", []) or [])
            # Avoid 2px-neighborhood matching and display from excluded_infos collected by the analyzer.
            # P numbers come from pre-filter peak indices, equivalent to init_peaks order, through an ID map.
            idx_map = getattr(best_model, "prefilter_index_by_id", None)
            excluded_rows: List[Tuple[Any, Dict[str, Any]]] = []
            for ent in excluded_infos:
                try:
                    pk = ent.get("peak")
                except Exception:
                    pk = None
                if pk is None:
                    continue
                idx = None
                try:
                    idx = ent.get("prefilter_index")
                except Exception:
                    idx = None
                if idx is None:
                    try:
                        idx = getattr(pk, "_prefilter_index", None)
                    except Exception:
                        idx = None
                try:
                    if isinstance(idx_map, dict):
                        idx = idx if idx is not None else idx_map.get(id(pk))
                except Exception:
                    idx = None
                try:
                    key = int(idx) if idx is not None else float("inf")
                    excluded_rows.append((key, ent))
                except Exception:
                    excluded_rows.append((float("inf"), ent))

            if excluded_rows:
                excluded_rows.sort(key=lambda t: (t[0], str(t[1].get("reasons", ""))))
                html_lines.append("")
                html_lines.append(esc("--- Excluded Peaks ---"))
                for idx0, ent in excluded_rows:
                    pk = ent.get("peak")
                    try:
                        reasons = [str(r) for r in list(ent.get("reasons") or []) if str(r).strip()]
                    except Exception:
                        reasons = []
                    if not reasons:
                        reasons = ["Failed to get reason"]
                    reason_str = ", ".join(reasons)
                    peak_label = f"P{idx0 + 1}" if np.isfinite(idx0) else "P?"
                    prefix = (
                        f"  [Excluded] {peak_label}: amp={pk.amplitude:.3g}, sigma={pk.sigma:.3g}, "
                        f"(x,y)=({pk.x:.2f},{pk.y:.2f}), S/N={pk.snr:.2f}  Reason: "
                    )
                    html_lines.append(
                        esc(prefix)
                        + '<span style="color:#c00; font-weight:600;">'
                        + esc(reason_str)
                        + "</span>"
                    )

        html_lines.append("")
        html_lines.append(esc(f"S/N Threshold (output filter): {result.snr_threshold:.2f}"))

        # Display as rich text (HTML), coloring only the reason text red.
        body = "\n".join(html_lines)
        html_doc = (
            '<pre style="margin:0; font-family:monospace; font-size:12px; white-space:pre-wrap;">'
            + body
            + "</pre>"
        )
        try:
            self.output.setHtml(html_doc)
        except Exception:
            # Fallback to plain text if needed
            try:
                self.output.setPlainText("\n".join([html.unescape(x) for x in html_lines]))
            except Exception:
                pass

    def _show_roi_view(self, result: FrameAnalysis) -> None:
        """Show or update the ROI visualization window"""
        # Do not use the ROI visualization window
        return

    def show_full_image_view(self, result: FrameAnalysis = None) -> None:
        """Overlay detected peaks on the full image; rectangle selection is also available"""
        if result is None:
            result = self.last_result
        # Show the current frame even without analysis results so ROI selection remains possible.
        if result is None or self.last_frame is None:
            if not self._ensure_selection_loaded():
                return
            frame = self._prepare_frame()
            if frame is None:
                QtWidgets.QMessageBox.information(self, "No Data", "No image is loaded.")
                return
            self.last_frame = frame
            result = None
        win = self._ensure_live_window(self.full_viz_window, SpotFullImageWindow)
        self.full_viz_window = win
        self.full_viz_window.enable_roi_selector(self.roi_shape_combo.currentText(), self._on_full_image_selected)
        self.full_viz_window.set_edit_handler(self._handle_edit_event)
        # Always pass both fit-spots and init-spots; rendering decides where to show them.
        spots = self.spots_by_frame.get(self._get_current_frame_index())
        initial_spots = self.initial_spots_by_frame.get(self._get_current_frame_index())
        self._apply_ui_to_analyzer(show_errors=False)
        _co, _rs, roi_mask, roi_bounds = self._roi_overrides(self.last_frame.shape)
        roi_pre, roi_det = self.spot_analyzer.compute_roi_visual_images(
            self.last_frame, roi_mask_override=roi_mask, roi_bounds_override=roi_bounds
        )
        win.update_view(
            self.last_frame,
            result,
            spots,
            initial_spots,
            self._current_roi_overlay(),
            self._get_spot_radius_px(),
            roi_pre_image=roi_pre,
            roi_det_image=roi_det,
            det_label=self.spot_analyzer.detection_label(),
            show_roi_spots=bool(getattr(self, "show_roi_spots_check", None) and self.show_roi_spots_check.isChecked()),
            show_det_spots=bool(getattr(self, "show_det_spots_check", None) and self.show_det_spots_check.isChecked()),
            show_initial_spots=False,
            show_fit_spots_on_roi_pre=bool(
                getattr(self, "show_fit_spots_on_roi_pre_check", None)
                and self.show_fit_spots_on_roi_pre_check.isChecked()
            ),
            init_mode=str(getattr(self.spot_analyzer, "init_mode", "") or ""),
        )
        win.show()
        win.raise_()
        win.activateWindow()

    def _get_spot_radius_px(self) -> int:
        try:
            return int(self.spot_radius_spin.value())
        except Exception:
            return 4

    def _on_spot_radius_changed(self, _value: int) -> None:
        # When radius changes, recompute current-frame heights and redraw.
        frame_index = self._get_current_frame_index()
        self._recompute_spot_heights_for_frame(frame_index)
        self._refresh_overlay()

    def _reset_analysis_results(self) -> None:
        """Clear analysis results, spots, ROI, UI state, and display state together."""
        # Clear data
        self.spots_by_frame = {}
        self.centroid_reference_by_frame = {}
        self.initial_spots_by_frame = {}
        self.roi_by_frame = {}
        self.manual_roi = None
        self.last_result = None
        self._analysis_signature_by_frame = {}
        try:
            self._reanalysis_timer.stop()
        except Exception:
            pass
        self._dragging = False
        self._drag_index = None

        # Restore the initial UI state
        try:
            try:
                self.output.setHtml("")
            except Exception:
                self.output.setPlainText("")
        except Exception:
            pass
        try:
            self.roi_status_label.setText("ROI Not Selected")
        except Exception:
            pass
        for btn in (getattr(self, "run_btn", None), getattr(self, "run_all_btn", None), getattr(self, "export_btn", None)):
            try:
                if btn is not None:
                    btn.setEnabled(False)
            except Exception:
                pass
        self._sync_run_buttons_enabled()

        # Update display by redrawing without spots/ROI
        self._refresh_overlay()

    def _on_full_image_selected(self, roi_info: Dict[str, float]) -> None:
        """ROI selection callback on the full image"""
        w = roi_info.get("w", 0)
        h = roi_info.get("h", 0)
        if w <= 1 or h <= 1:
            return
        frame_index = self._get_current_frame_index()
        self.roi_by_frame[frame_index] = roi_info
        self.manual_roi = roi_info
        self.roi_status_label.setText("ROI Selected")
        self._sync_run_buttons_enabled()
        # Reanalyze automatically using ROI center and size
        self.run_analysis()

    def _on_roi_shape_changed(self, text: str) -> None:
        if self.full_viz_window and self._is_window_live(self.full_viz_window):
            self.full_viz_window.enable_roi_selector(text, self._on_full_image_selected)

    def _roi_overrides(self, frame_shape: Tuple[int, int]):
        if self.manual_roi is None:
            return None, None, None, None
        h_img, w_img = frame_shape
        if self.manual_roi.get("shape") == "Ellipse":
            x0 = int(round(self.manual_roi["x0"]))
            y0 = int(round(self.manual_roi["y0"]))
            x1 = int(round(self.manual_roi["x0"] + self.manual_roi["w"]))
            y1 = int(round(self.manual_roi["y0"] + self.manual_roi["h"]))
            x0c = max(x0, 0)
            y0c = max(y0, 0)
            x1c = min(x1, w_img)
            y1c = min(y1, h_img)
            bounds = (x0c, y0c, x1c - x0c, y1c - y0c)
            cx = self.manual_roi["cx"]
            cy = self.manual_roi["cy"]
            rx = self.manual_roi["rx"]
            ry = self.manual_roi["ry"]
            roi_w = max(1, bounds[2])
            roi_h = max(1, bounds[3])
            yy, xx = np.mgrid[0:roi_h, 0:roi_w]
            mask = ((xx + bounds[0] - cx) ** 2) / (rx ** 2 + 1e-12) + ((yy + bounds[1] - cy) ** 2) / (ry ** 2 + 1e-12) <= 1.0
            return (cx, cy), None, mask, bounds
        # Rectangle
        x0 = int(round(self.manual_roi["x0"]))
        y0 = int(round(self.manual_roi["y0"]))
        x1 = int(round(self.manual_roi["x0"] + self.manual_roi["w"]))
        y1 = int(round(self.manual_roi["y0"] + self.manual_roi["h"]))
        x0c = max(x0, 0)
        y0c = max(y0, 0)
        x1c = min(x1, w_img)
        y1c = min(y1, h_img)
        bounds = (x0c, y0c, x1c - x0c, y1c - y0c)
        return None, None, None, bounds

    def _current_roi_overlay(self) -> Optional[Dict[str, float]]:
        return self.roi_by_frame.get(self._get_current_frame_index(), self.manual_roi)

    def _compute_circle_centroid_in_roi(
        self,
        roi_img: np.ndarray,
        x_local: float,
        y_local: float,
        radius_px: float,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Compute an intensity-weighted centroid inside a circular neighborhood.

        Weights are built from image values shifted by the local minimum so the
        centroid stays stable even when the image contains negative values.
        """
        img = np.asarray(roi_img, dtype=np.float64)
        if img.ndim != 2:
            return None
        h, w = img.shape
        try:
            cx = float(x_local)
            cy = float(y_local)
            r = float(radius_px)
        except Exception:
            return None
        if not np.isfinite(cx) or not np.isfinite(cy):
            return None
        if not np.isfinite(r) or r <= 0:
            return float(cx), float(cy)

        x_min = max(0, int(np.floor(cx - r)))
        x_max = min(w - 1, int(np.ceil(cx + r)))
        y_min = max(0, int(np.floor(cy - r)))
        y_max = min(h - 1, int(np.ceil(cy + r)))
        if x_min > x_max or y_min > y_max:
            return float(cx), float(cy)

        yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(r) ** 2
        if roi_mask is not None:
            try:
                local_mask = roi_mask[y_min : y_max + 1, x_min : x_max + 1].astype(bool)
                circle &= local_mask
            except Exception:
                pass
        if not np.any(circle):
            return float(cx), float(cy)

        values = np.asarray(img[y_min : y_max + 1, x_min : x_max + 1], dtype=np.float64)
        finite_circle = circle & np.isfinite(values)
        if not np.any(finite_circle):
            return float(cx), float(cy)

        vals = values[finite_circle]
        floor_v = float(np.min(vals))
        weights = np.where(finite_circle, values - floor_v, 0.0)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return float(cx), float(cy)

        x_cent = float(np.sum(xx * weights) / total)
        y_cent = float(np.sum(yy * weights) / total)
        return x_cent, y_cent

    def _relabel_result_peaks_by_previous_frame(self, frame_index: int, result: Optional[FrameAnalysis]) -> None:
        """
        Reorder the selected model's final peaks so their labels follow the nearest
        peaks from the immediately previous analyzed frame.

        - If frame_index-1 has no stored peaks, keep the current order.
        - Matched peaks inherit the previous frame's label order.
        - Newly appearing peaks are appended in their original order.
        """
        if result is None or result.best_n_peaks is None:
            return
        try:
            best_model = result.models[result.best_n_peaks]
        except Exception:
            return

        prev_spots = list(self.spots_by_frame.get(int(frame_index) - 1) or [])
        curr_peaks = list(getattr(best_model, "peaks", []) or [])
        if not prev_spots or len(curr_peaks) <= 1:
            return

        prev_xy: List[Tuple[float, float]] = []
        for pk in prev_spots:
            try:
                prev_xy.append((float(pk.get("x", float("nan"))), float(pk.get("y", float("nan")))))
            except Exception:
                prev_xy.append((float("nan"), float("nan")))
        if not prev_xy or not any(np.isfinite(x) and np.isfinite(y) for x, y in prev_xy):
            return

        curr_xy = np.array([[float(pk.x), float(pk.y)] for pk in curr_peaks], dtype=np.float64)
        prev_xy_arr = np.array(prev_xy, dtype=np.float64)
        if curr_xy.ndim != 2 or prev_xy_arr.ndim != 2:
            return

        dist = np.linalg.norm(prev_xy_arr[:, None, :] - curr_xy[None, :, :], axis=2)
        if not np.all(np.isfinite(dist)):
            dist = np.where(np.isfinite(dist), dist, 1e9)

        matched_curr_by_prev: Dict[int, int] = {}
        try:
            row_ind, col_ind = linear_sum_assignment(dist)
            pairs = sorted(zip(row_ind.tolist(), col_ind.tolist()), key=lambda t: t[0])
            for prev_i, curr_i in pairs:
                if 0 <= prev_i < len(prev_spots) and 0 <= curr_i < len(curr_peaks):
                    matched_curr_by_prev[int(prev_i)] = int(curr_i)
        except Exception:
            used_curr: set[int] = set()
            for prev_i in range(len(prev_spots)):
                best_j = None
                best_d = None
                for curr_i in range(len(curr_peaks)):
                    if curr_i in used_curr:
                        continue
                    d = float(dist[prev_i, curr_i])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = curr_i
                if best_j is not None:
                    matched_curr_by_prev[int(prev_i)] = int(best_j)
                    used_curr.add(int(best_j))

        ordered_indices: List[int] = []
        used_indices: set[int] = set()
        for prev_i in range(len(prev_spots)):
            curr_i = matched_curr_by_prev.get(int(prev_i))
            if curr_i is None or curr_i in used_indices:
                continue
            ordered_indices.append(int(curr_i))
            used_indices.add(int(curr_i))
        for curr_i in range(len(curr_peaks)):
            if curr_i not in used_indices:
                ordered_indices.append(int(curr_i))

        if len(ordered_indices) != len(curr_peaks):
            return

        best_model.peaks = [curr_peaks[i] for i in ordered_indices]

    def _store_spots_for_frame(
        self,
        frame_index: int,
        result: FrameAnalysis,
        frame: Optional[np.ndarray] = None,
        update_initial: bool = True,
    ) -> None:
        best = result.models[result.best_n_peaks]
        self.spots_by_frame[frame_index] = [
            {"x": float(pk.x), "y": float(pk.y), "snr": float(pk.snr)}
            for pk in best.peaks
        ]
        self._sync_centroid_reference_for_frame(frame_index)
        if update_initial:
            # Display initial positions as detection seeds (independent of best_n_peaks / initial_sigma)
            # Fallback to legacy init_peaks if seeds are not available.
            if getattr(result, "seed_spots", None):
                self.initial_spots_by_frame[frame_index] = [
                    {"x": float(pk.get("x", 0.0)), "y": float(pk.get("y", 0.0)), "snr": float(pk.get("snr", 0.0))}
                    for pk in (result.seed_spots or [])
                ]
            else:
                self.initial_spots_by_frame[frame_index] = [
                    {"x": float(pk.x), "y": float(pk.y), "snr": float(pk.snr)}
                    for pk in best.init_peaks
                ]
        # Add height information
        use_frame = frame if frame is not None else self.last_frame
        if use_frame is not None:
            self._recompute_spot_heights_for_frame(frame_index, frame=use_frame)

    def _sync_centroid_reference_for_frame(
        self,
        frame_index: int,
        spots: Optional[Sequence[Dict[str, float]]] = None,
    ) -> None:
        src = list(spots) if spots is not None else list(self.spots_by_frame.get(frame_index) or [])
        self.centroid_reference_by_frame[frame_index] = [dict(pk) for pk in src]

    def _refresh_overlay(self) -> None:
        if not self.full_viz_window or not self._is_window_live(self.full_viz_window):
            return
        frame_index = self._get_current_frame_index()
        # Always pass both fit-spots and init-spots; rendering decides where to show them.
        spots = self.spots_by_frame.get(frame_index)
        initial_spots = self.initial_spots_by_frame.get(frame_index)
        if self.last_frame is None:
            return
        self._apply_ui_to_analyzer(show_errors=False)
        _co, _rs, roi_mask, roi_bounds = self._roi_overrides(self.last_frame.shape)
        roi_pre, roi_det = self.spot_analyzer.compute_roi_visual_images(
            self.last_frame, roi_mask_override=roi_mask, roi_bounds_override=roi_bounds
        )
        self.full_viz_window.update_view(
            self.last_frame,
            self.last_result,
            spots,
            initial_spots,
            self._current_roi_overlay(),
            self._get_spot_radius_px(),
            roi_pre_image=roi_pre,
            roi_det_image=roi_det,
            det_label=self.spot_analyzer.detection_label(),
            show_roi_spots=bool(getattr(self, "show_roi_spots_check", None) and self.show_roi_spots_check.isChecked()),
            show_det_spots=bool(getattr(self, "show_det_spots_check", None) and self.show_det_spots_check.isChecked()),
            show_initial_spots=False,
            show_fit_spots_on_roi_pre=bool(
                getattr(self, "show_fit_spots_on_roi_pre_check", None)
                and self.show_fit_spots_on_roi_pre_check.isChecked()
            ),
            init_mode=str(getattr(self.spot_analyzer, "init_mode", "") or ""),
        )

    def _get_roi_info_for_frame(self, frame_index: int) -> Optional[Dict[str, float]]:
        return self.roi_by_frame.get(frame_index, self.manual_roi)

    def _compute_spot_circle_mean_nm(self, frame: np.ndarray, x: float, y: float, r_px: int) -> Optional[float]:
        if r_px <= 0:
            return None
        h, w = frame.shape
        x0 = max(int(np.floor(x - r_px)), 0)
        x1 = min(int(np.ceil(x + r_px)) + 1, w)
        y0 = max(int(np.floor(y - r_px)), 0)
        y1 = min(int(np.ceil(y + r_px)) + 1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= float(r_px) ** 2
        data = frame[y0:y1, x0:x1][mask]
        if data.size == 0:
            return None
        return float(np.mean(data))

    def _compute_spot_point_height_nm(self, frame: np.ndarray, x: float, y: float) -> Optional[float]:
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None
        h, w = img.shape
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            return None
        if not np.isfinite(xf) or not np.isfinite(yf):
            return None
        if xf < 0.0 or yf < 0.0 or xf > float(w - 1) or yf > float(h - 1):
            return None

        x0 = int(np.floor(xf))
        y0 = int(np.floor(yf))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        dx = float(xf - x0)
        dy = float(yf - y0)

        v00 = float(img[y0, x0])
        v10 = float(img[y0, x1])
        v01 = float(img[y1, x0])
        v11 = float(img[y1, x1])
        if not np.all(np.isfinite([v00, v10, v01, v11])):
            return None

        v0 = v00 * (1.0 - dx) + v10 * dx
        v1 = v01 * (1.0 - dx) + v11 * dx
        return float(v0 * (1.0 - dy) + v1 * dy)

    def _selected_height_export_mode(self) -> str:
        try:
            text = str(self.height_export_mode_combo.currentText() or "").strip()
        except Exception:
            text = ""
        if text == "Spot Position":
            return "point"
        return "mean"

    def _compute_background_median_nm(
        self,
        frame: np.ndarray,
        frame_index: int,
        r_px: int,
        spots: Sequence[Dict[str, float]],
    ) -> float:
        """
        Background = median of the full frame.

        Arguments are kept for compatibility, but the current calculation does not depend on ROI or spot placement.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.size == 0:
            return 0.0
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return 0.0
        return float(np.median(finite))

    def _recompute_spot_heights_for_frame(self, frame_index: int, frame: Optional[np.ndarray] = None) -> None:
        spots = self.spots_by_frame.get(frame_index)
        if not spots:
            return
        use_frame = frame if frame is not None else self.last_frame
        if use_frame is None:
            return
        r_px = self._get_spot_radius_px()
        bg_nm = self._compute_background_median_nm(use_frame, frame_index, r_px, spots)
        for pk in spots:
            point_nm = self._compute_spot_point_height_nm(use_frame, float(pk["x"]), float(pk["y"]))
            mean_nm = self._compute_spot_circle_mean_nm(use_frame, float(pk["x"]), float(pk["y"]), r_px)
            if point_nm is None and mean_nm is None:
                continue
            pk["height_bg_nm"] = float(bg_nm)
            if point_nm is not None:
                pk["height_point_nm"] = float(point_nm)
                pk["height_point_bgsub_nm"] = float(point_nm - bg_nm)
            if mean_nm is not None:
                pk["height_mean_nm"] = float(mean_nm)
                pk["height_mean_bgsub_nm"] = float(mean_nm - bg_nm)

            # Backward-compatible primary fields. Keep them aligned to the
            # currently selected export mode so old consumers continue to work.
            mode = self._selected_height_export_mode()
            if mode == "point" and point_nm is not None:
                pk["height_bgsub_nm"] = float(point_nm - bg_nm)
            elif mean_nm is not None:
                pk["height_bgsub_nm"] = float(mean_nm - bg_nm)
            elif point_nm is not None:
                pk["height_bgsub_nm"] = float(point_nm - bg_nm)

    def _recompute_spot_heights_for_all_frames(self) -> None:
        if not self.spots_by_frame:
            return
        if not self._ensure_selection_loaded():
            return

        original_index = int(getattr(gv, "index", 0))
        try:
            for frame_index in sorted(self.spots_by_frame.keys()):
                spots = self.spots_by_frame.get(frame_index)
                if not spots:
                    continue
                gv.index = int(frame_index)
                frame = self._prepare_frame()
                if frame is None:
                    continue
                self.last_frame = frame
                self._recompute_spot_heights_for_frame(int(frame_index), frame=frame)
        finally:
            gv.index = original_index
            frame = self._prepare_frame()
            if frame is not None:
                self.last_frame = frame
            self._refresh_overlay()

    def _handle_edit_event(self, event, phase: str) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        frame_index = self._get_current_frame_index()
        spots = self.spots_by_frame.setdefault(frame_index, [])
        key = (event.key or "").lower()
        is_shift = "shift" in key
        is_delete = "alt" in key or "option" in key
        is_ctrl = "control" in key or "ctrl" in key
        is_cmd = "cmd" in key or "command" in key or "meta" in key or "super" in key
        if not key:
            mods = QtWidgets.QApplication.keyboardModifiers()
            is_shift = bool(mods & QtCore.Qt.ShiftModifier)
            is_delete = bool(mods & QtCore.Qt.AltModifier)
            is_ctrl = bool(mods & QtCore.Qt.ControlModifier)
            is_cmd = bool(mods & QtCore.Qt.MetaModifier)
        # In the Matplotlib Qt backend, modifier keys may not appear in event.key or keyboardModifiers, so
        # also read them from the original Qt event to handle macOS Ctrl+click/drag as right-click, etc.
        try:
            gui_ev = getattr(event, "guiEvent", None)
            if gui_ev is not None and hasattr(gui_ev, "modifiers"):
                is_ctrl = is_ctrl or bool(gui_ev.modifiers() & QtCore.Qt.ControlModifier)
                is_cmd = is_cmd or bool(gui_ev.modifiers() & QtCore.Qt.MetaModifier)
        except Exception:
            pass

        if phase == "press":
            if is_shift:
                spots.append({"x": float(event.xdata), "y": float(event.ydata), "snr": 0.0})
                self._sync_centroid_reference_for_frame(frame_index, spots)
                self.export_btn.setEnabled(True)
                self._recompute_spot_heights_for_frame(frame_index)
                self._refresh_overlay()
                return
            if is_delete:
                idx = self._find_nearest_spot(spots, event.xdata, event.ydata)
                if idx is not None:
                    spots.pop(idx)
                    self._sync_centroid_reference_for_frame(frame_index, spots)
                    self.export_btn.setEnabled(True)
                    self._recompute_spot_heights_for_frame(frame_index)
                    self._refresh_overlay()
                return
            # Start moving only while Ctrl or Cmd is pressed to avoid conflicting with ROI drawing drags.
            if not (is_ctrl or is_cmd):
                return
            idx = self._find_nearest_spot(spots, event.xdata, event.ydata)
            if idx is not None:
                self._dragging = True
                self._drag_index = idx
        elif phase == "move":
            if self._dragging and self._drag_index is not None:
                spots[self._drag_index]["x"] = float(event.xdata)
                spots[self._drag_index]["y"] = float(event.ydata)
                self._sync_centroid_reference_for_frame(frame_index, spots)
                self._recompute_spot_heights_for_frame(frame_index)
                self._refresh_overlay()
        elif phase == "release":
            if self._dragging:
                self._dragging = False
                self._drag_index = None
                self._sync_centroid_reference_for_frame(frame_index, spots)
                self.export_btn.setEnabled(True)
                self._recompute_spot_heights_for_frame(frame_index)

    def _find_nearest_spot(self, spots: List[Dict[str, float]], x: float, y: float, threshold: float = 5.0) -> Optional[int]:
        if not spots:
            return None
        best_idx = None
        best_dist = None
        for i, pk in enumerate(spots):
            dx = pk["x"] - x
            dy = pk["y"] - y
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist is not None and best_dist <= threshold:
            return best_idx
        return None

    def run_analysis_all_frames(self) -> None:
        if hasattr(self, "run_all_enable_check") and not self.run_all_enable_check.isChecked():
            QtWidgets.QMessageBox.information(self, "All Frames Disabled", "Turn ON Enable All Frames Analysis first.")
            return
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "Select an ROI first.")
            return
        if not self._ensure_selection_loaded():
            return
        if not hasattr(gv, "FrameNum") or gv.FrameNum <= 0:
            QtWidgets.QMessageBox.warning(self, "No Frames", "Could not get the number of frames.")
            return
        if not self._apply_ui_to_analyzer(show_errors=True):
            return
        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "The maximum peak count must be greater than or equal to the minimum peak count.")
            return
        original_index = int(getattr(gv, "index", 0))
        self.spots_by_frame = {}
        self.centroid_reference_by_frame = {}
        self.initial_spots_by_frame = {}
        self._analysis_signature_by_frame = {}
        for idx in range(int(gv.FrameNum)):
            gv.index = idx
            frame = self._prepare_frame()
            if frame is None:
                continue
            self.last_frame = frame
            center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
            try:
                result = self.spot_analyzer.analyze_frame(
                    frame,
                    prev_center=None,
                    criterion=self.criterion_combo.currentText().lower(),
                    center_override=center_override,
                    roi_size_override=roi_size_override,
                    roi_mask_override=roi_mask,
                    roi_bounds_override=roi_bounds,
                    min_peaks=min_peaks,
                    max_peaks=max_peaks,
                )
            except Exception:
                continue
            self._relabel_result_peaks_by_previous_frame(idx, result)
            self.last_result = result
            self._store_spots_for_frame(idx, result, frame=frame)
            if getattr(self, "auto_centroid_check", None) is not None and self.auto_centroid_check.isChecked():
                self._apply_spot_centroid_to_frame(idx, frame=frame, result=result, show_messages=False, refresh_display=False)
            # Record signature to skip reanalysis under the same conditions
            try:
                roi_info = self.roi_by_frame.get(idx, self.manual_roi)
                self._analysis_signature_by_frame[idx] = self._analysis_signature(idx, roi_info)
            except Exception:
                pass
        gv.index = original_index
        frame = self._prepare_frame()
        if frame is not None:
            self.last_frame = frame
        self.export_btn.setEnabled(True)
        self._refresh_overlay()

    def _analyze_current_frame(
        self,
        frame: np.ndarray,
        criterion: str,
        center_override: Optional[Tuple[float, float]],
        roi_size_override: Optional[int],
        roi_mask: Optional[np.ndarray],
        roi_bounds: Optional[Tuple[int, int, int, int]],
        min_peaks: int,
        max_peaks: int,
        show_errors: bool = False,
    ) -> None:
        try:
            result = self.spot_analyzer.analyze_frame(
                frame,
                prev_center=None,
                criterion=criterion,
                center_override=center_override,
                roi_size_override=roi_size_override,
                roi_mask_override=roi_mask,
                roi_bounds_override=roi_bounds,
                min_peaks=min_peaks,
                max_peaks=max_peaks,
            )
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "Analysis Error", f"Failed to run SpotAnalysis:\n{exc}")
            return

        self.last_frame = frame
        self._relabel_result_peaks_by_previous_frame(self._get_current_frame_index(), result)
        self.last_result = result
        frame_index = self._get_current_frame_index()
        self._store_spots_for_frame(frame_index, result, frame=frame)
        if getattr(self, "auto_centroid_check", None) is not None and self.auto_centroid_check.isChecked():
            self._apply_spot_centroid_to_frame(
                frame_index,
                frame=frame,
                result=result,
                show_messages=False,
                refresh_display=False,
            )
        self._display_result(result)
        # Record signature so reanalysis can be skipped when conditions are unchanged
        try:
            fi = frame_index
            self._analysis_signature_by_frame[fi] = self._analysis_signature(fi, self._current_roi_overlay())
        except Exception:
            pass
        self._refresh_overlay()
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _auto_analyze_if_enabled(self, frame_index: int, frame: np.ndarray) -> None:
        if not self.auto_analyze_check.isChecked():
            return
        if self._auto_busy:
            return
        if self.manual_roi is None:
            return
        if not self._apply_ui_to_analyzer(show_errors=False):
            return
        if self._should_skip_auto_analysis_for_frame(frame_index):
            return

        roi_info = self._current_roi_overlay()
        sig = self._analysis_signature(frame_index, roi_info)
        if self._analysis_signature_by_frame.get(frame_index) == sig:
            return

        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            return

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
        self._auto_busy = True
        try:
            self._analyze_current_frame(
                frame,
                self.criterion_combo.currentText().lower(),
                center_override,
                roi_size_override,
                roi_mask,
                roi_bounds,
                min_peaks,
                max_peaks,
                show_errors=False,
            )
        finally:
            self._auto_busy = False

    def export_spots_csv(self) -> None:
        if not self.spots_by_frame:
            QtWidgets.QMessageBox.information(self, "No Data", "There are no spots to save.")
            return
        if not hasattr(gv, "files") or not gv.files or gv.currentFileNum < 0:
            QtWidgets.QMessageBox.warning(self, "No File", "Could not get source file information.")
            return
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize") or gv.XScanSize == 0 or gv.YScanSize == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Scan Size", "scan_size is 0. Load valid data first.")
            return
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel") or gv.XPixel == 0 or gv.YPixel == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Pixel Size", "Could not get pixel-size information.")
            return

        nm_per_pixel_x = gv.XScanSize / gv.XPixel
        nm_per_pixel_y = gv.YScanSize / gv.YPixel

        src_path = gv.files[gv.currentFileNum]
        base_dir = os.path.dirname(src_path)
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        if self.export_dir is None:
            self.export_dir = base_dir
        default_path = os.path.join(self.export_dir, f"{base_name}.csv")
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save CSV (base name)",
            default_path,
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return
        export_dir = os.path.dirname(selected_path)
        base_stub = os.path.splitext(os.path.basename(selected_path))[0] or base_name
        self.export_dir = export_dir
        export_height_mode = self._selected_height_export_mode()

        # Export always refreshes heights from the current spot positions so
        # CSV-restored spot coordinates can be reused to regenerate heights.
        self._recompute_spot_heights_for_all_frames()

        # save meta.json alongside CSVs
        try:
            self._save_spot_meta_json(os.path.join(export_dir, f"{base_stub}_meta.json"), nm_per_pixel_x, nm_per_pixel_y)
        except Exception:
            pass

        max_spots = max(len(v) for v in self.spots_by_frame.values())
        # pre-check conflicts and ask once
        conflict = False
        for p in range(1, int(max_spots) + 1):
            pos_path = os.path.join(export_dir, f"{base_stub}_p{p}.csv")
            h_path = os.path.join(export_dir, f"{base_stub}_p{p}_h.csv")
            if os.path.exists(pos_path) or os.path.exists(h_path):
                conflict = True
                break

        mode = "number"
        if conflict:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("File Exists")
            box.setText("CSV files with the same name already exist. What would you like to do?")
            overwrite_btn = box.addButton("Overwrite", QtWidgets.QMessageBox.AcceptRole)
            number_btn = box.addButton("Create Numbered Files", QtWidgets.QMessageBox.ActionRole)
            cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            box.setDefaultButton(number_btn)
            box.exec_()
            clicked = box.clickedButton()
            if clicked == cancel_btn:
                return
            if clicked == overwrite_btn:
                mode = "overwrite"
            else:
                mode = "number"

        for spot_index in range(max_spots):
            out_pos_path, out_h_path = self._unique_spot_paths(export_dir, base_stub, spot_index + 1, mode=mode)
            if out_pos_path is None or out_h_path is None:
                QtWidgets.QMessageBox.warning(self, "Save Error", "Failed to resolve output filename conflicts.")
                return
            try:
                # Position CSV
                with open(out_pos_path, "w", encoding="utf-8") as f_pos:
                    f_pos.write("frame_index,x_nm,y_nm\n")
                    for frame_idx in sorted(self.spots_by_frame.keys()):
                        spots = self.spots_by_frame[frame_idx]
                        if spot_index >= len(spots):
                            continue
                        x_nm = spots[spot_index]["x"] * nm_per_pixel_x
                        y_nm = spots[spot_index]["y"] * nm_per_pixel_y

                        f_pos.write(f"{frame_idx},{x_nm:.6f},{y_nm:.6f}\n")

                # Height CSV
                with open(out_h_path, "w", encoding="utf-8") as f_h:
                    f_h.write(
                        "frame_index,height_value_nm,height_bg_nm,height_bgsub_nm,height_mode,"
                        "height_point_nm,height_point_bgsub_nm,height_mean_nm,height_mean_bgsub_nm\n"
                    )
                    for frame_idx in sorted(self.spots_by_frame.keys()):
                        spots = self.spots_by_frame[frame_idx]
                        if spot_index >= len(spots):
                            continue
                        # If height is not calculated, calculate from current frame data when possible
                        if (
                            "height_point_nm" not in spots[spot_index]
                            or "height_mean_nm" not in spots[spot_index]
                            or "height_bgsub_nm" not in spots[spot_index]
                            or "height_bg_nm" not in spots[spot_index]
                        ):
                            if self.last_frame is not None and frame_idx == self._get_current_frame_index():
                                self._recompute_spot_heights_for_frame(frame_idx, frame=self.last_frame)
                        h_point = float(spots[spot_index].get("height_point_nm", float("nan")))
                        h_point_bgsub = float(spots[spot_index].get("height_point_bgsub_nm", float("nan")))
                        h_mean = float(spots[spot_index].get("height_mean_nm", float("nan")))
                        h_mean_bgsub = float(spots[spot_index].get("height_mean_bgsub_nm", float("nan")))
                        h_bg = float(spots[spot_index].get("height_bg_nm", float("nan")))
                        if export_height_mode == "point":
                            h_value = h_point
                            h_bgsub = h_point_bgsub
                            h_mode = "point"
                        else:
                            h_value = h_mean
                            h_bgsub = h_mean_bgsub
                            h_mode = "mean"
                        f_h.write(
                            f"{frame_idx},{h_value:.6f},{h_bg:.6f},{h_bgsub:.6f},{h_mode},"
                            f"{h_point:.6f},{h_point_bgsub:.6f},{h_mean:.6f},{h_mean_bgsub:.6f}\n"
                        )
            except Exception:
                continue

    def _unique_spot_paths(
        self, export_dir: str, base_stub: str, p_index: int, mode: str = "number", max_tries: int = 999
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate non-existing paired paths:
          <base>_pN.csv and <base>_pN_h.csv
        If any exists, append _k suffix before extension:
          <base>_pN_1.csv and <base>_pN_h_1.csv
        """
        p = int(p_index)
        if str(mode).strip().lower() == "overwrite":
            pos_path = os.path.join(export_dir, f"{base_stub}_p{p}.csv")
            h_path = os.path.join(export_dir, f"{base_stub}_p{p}_h.csv")
            return pos_path, h_path
        for k in range(int(max_tries) + 1):
            suffix = "" if k == 0 else f"_{k}"
            pos_name = f"{base_stub}_p{p}{suffix}.csv"
            h_name = f"{base_stub}_p{p}_h{suffix}.csv"
            pos_path = os.path.join(export_dir, pos_name)
            h_path = os.path.join(export_dir, h_name)
            if not os.path.exists(pos_path) and not os.path.exists(h_path):
                return pos_path, h_path
        return None, None

    def _save_spot_meta_json(self, path: str, nm_per_pixel_x: float, nm_per_pixel_y: float) -> None:
        ui_params = self._collect_ui_params()
        meta = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frame_num": int(getattr(gv, "FrameNum", 0) or 0),
            "nm_per_pixel_x": float(nm_per_pixel_x),
            "nm_per_pixel_y": float(nm_per_pixel_y),
            "manual_roi": self.manual_roi,
            "roi_by_frame": self.roi_by_frame,
            "ui_params": ui_params,
            "analysis_signature_by_frame": {str(k): v for k, v in self._analysis_signature_by_frame.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _collect_ui_params(self) -> Dict[str, object]:
        """Collect UI values needed to reproduce analysis signature/behavior."""
        def _try_get(getter, default=None):
            try:
                return getter()
            except Exception:
                return default

        return {
            "criterion": _try_get(lambda: (self.criterion_combo.currentText() or "AIC")),
            "detection_mode": _try_get(lambda: (self.detection_mode_combo.currentText() or "DoG")),
            "bandpass_low_sigma": _try_get(lambda: float(self.bandpass_low_spin.value()), 0.6),
            "bandpass_high_sigma": _try_get(lambda: float(self.bandpass_high_spin.value()), 3.0),
            "log_sigma": _try_get(lambda: float(self.log_sigma_spin.value()), 1.6),
            "median_enabled": _try_get(lambda: bool(self.median_check.isChecked()), False),
            "median_size": _try_get(lambda: int(self.median_size_spin.value()), 3),
            "open_enabled": _try_get(lambda: bool(self.open_check.isChecked()), False),
            "open_radius": _try_get(lambda: int(self.open_radius_spin.value()), 1),
            "init_mode": _try_get(lambda: (self.init_mode_combo.currentText() or "Watershed (Recommended)")),
            "subpixel_refine": _try_get(lambda: bool(self.subpixel_check.isChecked()), False),
            "watershed_h_rel": _try_get(lambda: float(self.watershed_h_rel_spin.value()), 0.05),
            "watershed_adaptive_h": _try_get(lambda: bool(self.watershed_adaptive_h_check.isChecked()), True),
            "blob_doh_min_sigma": _try_get(lambda: float(self.blob_doh_min_sigma_spin.value()), 3.0),
            "blob_doh_max_sigma": _try_get(lambda: float(self.blob_doh_max_sigma_spin.value()), 20.0),
            "blob_doh_threshold_rel": _try_get(lambda: float(self.blob_doh_threshold_rel_spin.value()), 0.01),
            "dbscan_enabled": _try_get(lambda: bool(self.dbscan_enabled_check.isChecked()), False),
            "dbscan_eps": _try_get(lambda: float(self.dbscan_eps_spin.value()), 5.0),
            "min_peaks": _try_get(lambda: int(self.min_peaks_spin.value()), 1),
            "max_peaks": _try_get(lambda: int(self.max_peaks_spin.value()), 3),
            "snr_threshold": _try_get(lambda: float(self.snr_spin.value()), 2.0),
            "peak_min_distance": _try_get(lambda: int(self.peak_min_distance_spin.value()), 2),
            "localmax_threshold_rel": _try_get(lambda: float(self.localmax_thr_spin.value()), 0.05),
            "localmax_threshold_snr": _try_get(lambda: float(self.localmax_snr_spin.value()), 0.0),
            "precheck_radius_px": _try_get(lambda: int(self.precheck_radius_spin.value()), 2),
            "precheck_kmad": _try_get(lambda: float(self.precheck_kmad_spin.value()), 1.0),
            "fit_enabled": _try_get(lambda: bool(self.fit_enabled_check.isChecked()), True),
            "auto_centroid": _try_get(lambda: bool(self.auto_centroid_check.isChecked()), False),
            "initial_sigma": _try_get(lambda: float(self.initial_sigma_spin.value()), 2.0),
            "sigma_min": _try_get(lambda: float(self.sigma_min_spin.value()), 0.6),
            "sigma_max": _try_get(lambda: float(self.sigma_max_spin.value()), 8.0),
            "margin": _try_get(lambda: int(self.margin_spin.value()), 0),
            "min_amplitude": _try_get(lambda: float(self.min_amp_spin.value()), 0.0),
            "min_sigma_result": _try_get(lambda: float(self.min_sigma_result_spin.value()), 0.0),
            "snap_enabled": _try_get(lambda: bool(self.snap_enabled_check.isChecked()), False),
            "snap_radius": _try_get(lambda: int(self.snap_radius_spin.value()), 2),
            "snap_refit_enabled": _try_get(lambda: bool(self.snap_refit_check.isChecked()), False),
            "height_export_mode": _try_get(self._selected_height_export_mode, "mean"),
        }

    def _apply_ui_params(self, params: Dict[str, object]) -> None:
        """Best-effort apply UI params without triggering reanalysis repeatedly."""
        blockers = []
        for w in (
            getattr(self, "criterion_combo", None),
            getattr(self, "detection_mode_combo", None),
            getattr(self, "bandpass_low_spin", None),
            getattr(self, "bandpass_high_spin", None),
            getattr(self, "log_sigma_spin", None),
            getattr(self, "median_check", None),
            getattr(self, "median_size_spin", None),
            getattr(self, "open_check", None),
            getattr(self, "open_radius_spin", None),
            getattr(self, "init_mode_combo", None),
            getattr(self, "subpixel_check", None),
            getattr(self, "watershed_h_rel_spin", None),
            getattr(self, "watershed_adaptive_h_check", None),
            getattr(self, "blob_min_sigma_spin", None),
            getattr(self, "blob_max_sigma_spin", None),
            getattr(self, "blob_num_sigma_spin", None),
            getattr(self, "blob_threshold_rel_spin", None),
            getattr(self, "blob_overlap_spin", None),
            getattr(self, "blob_doh_min_sigma_spin", None),
            getattr(self, "blob_doh_max_sigma_spin", None),
            getattr(self, "blob_doh_threshold_rel_spin", None),
            getattr(self, "dbscan_enabled_check", None),
            getattr(self, "dbscan_eps_spin", None),
            getattr(self, "min_peaks_spin", None),
            getattr(self, "max_peaks_spin", None),
            getattr(self, "snr_spin", None),
            getattr(self, "peak_min_distance_spin", None),
            getattr(self, "localmax_thr_spin", None),
            getattr(self, "localmax_snr_spin", None),
            getattr(self, "precheck_radius_spin", None),
            getattr(self, "precheck_kmad_spin", None),
            getattr(self, "fit_enabled_check", None),
            getattr(self, "auto_centroid_check", None),
            getattr(self, "initial_sigma_spin", None),
            getattr(self, "sigma_min_spin", None),
            getattr(self, "sigma_max_spin", None),
            getattr(self, "margin_spin", None),
            getattr(self, "min_amp_spin", None),
            getattr(self, "min_sigma_result_spin", None),
            getattr(self, "snap_enabled_check", None),
            getattr(self, "snap_radius_spin", None),
            getattr(self, "snap_refit_check", None),
            getattr(self, "height_export_mode_combo", None),
        ):
            try:
                if w is not None:
                    blockers.append(QtCore.QSignalBlocker(w))
            except Exception:
                pass

        try:
            if "criterion" in params:
                self.criterion_combo.setCurrentText(str(params["criterion"]))
            if "detection_mode" in params:
                det = str(params["detection_mode"]).strip().lower()
                if det in ("pre", "preprocessed"):
                    self.detection_mode_combo.setCurrentText("Preprocessed")
                elif det == "log":
                    self.detection_mode_combo.setCurrentText("LoG")
                elif det == "dog":
                    self.detection_mode_combo.setCurrentText("DoG")
                else:
                    self.detection_mode_combo.setCurrentText(str(params["detection_mode"]))
            if "bandpass_low_sigma" in params:
                self.bandpass_low_spin.setValue(float(params["bandpass_low_sigma"]))
            if "bandpass_high_sigma" in params:
                self.bandpass_high_spin.setValue(float(params["bandpass_high_sigma"]))
            if "log_sigma" in params:
                self.log_sigma_spin.setValue(float(params["log_sigma"]))
            if "median_enabled" in params:
                self.median_check.setChecked(bool(params["median_enabled"]))
            if "median_size" in params:
                self.median_size_spin.setValue(int(params["median_size"]))
            if "open_enabled" in params:
                self.open_check.setChecked(bool(params["open_enabled"]))
            if "open_radius" in params:
                self.open_radius_spin.setValue(int(params["open_radius"]))
            if "init_mode" in params:
                # Map old mode names to new UI labels
                old_mode = str(params["init_mode"]).lower()
                if "watershed" in old_mode or "water" in old_mode:
                    self.init_mode_combo.setCurrentText("Watershed (Recommended)")
                elif "doh" in old_mode:
                    self.init_mode_combo.setCurrentText("Blob DoH (Fast)")
                elif "blob" in old_mode:
                    # Old blob_log maps to blob_doh
                    self.init_mode_combo.setCurrentText("Blob DoH (Fast)")
                elif "peak" in old_mode or "multiscale" in old_mode:
                    self.init_mode_combo.setCurrentText("Peak")
                else:
                    # Try to set directly if it matches new format
                    self.init_mode_combo.setCurrentText(str(params["init_mode"]))
            if "subpixel_refine" in params:
                self.subpixel_check.setChecked(bool(params["subpixel_refine"]))
            if "watershed_h_rel" in params:
                self.watershed_h_rel_spin.setValue(float(params["watershed_h_rel"]))
            if "watershed_adaptive_h" in params:
                self.watershed_adaptive_h_check.setChecked(bool(params["watershed_adaptive_h"]))
            if "blob_doh_min_sigma" in params:
                self.blob_doh_min_sigma_spin.setValue(float(params["blob_doh_min_sigma"]))
            if "blob_doh_max_sigma" in params:
                self.blob_doh_max_sigma_spin.setValue(float(params["blob_doh_max_sigma"]))
            if "blob_doh_threshold_rel" in params:
                self.blob_doh_threshold_rel_spin.setValue(float(params["blob_doh_threshold_rel"]))
            if "dbscan_enabled" in params:
                self.dbscan_enabled_check.setChecked(bool(params["dbscan_enabled"]))
            if "dbscan_eps" in params:
                self.dbscan_eps_spin.setValue(float(params["dbscan_eps"]))
            if "min_peaks" in params:
                self.min_peaks_spin.setValue(int(params["min_peaks"]))
            if "max_peaks" in params:
                self.max_peaks_spin.setValue(int(params["max_peaks"]))
            if "snr_threshold" in params:
                self.snr_spin.setValue(float(params["snr_threshold"]))
            if "peak_min_distance" in params:
                self.peak_min_distance_spin.setValue(int(params["peak_min_distance"]))
            if "localmax_threshold_rel" in params:
                self.localmax_thr_spin.setValue(float(params["localmax_threshold_rel"]))
            if "localmax_threshold_snr" in params:
                self.localmax_snr_spin.setValue(float(params["localmax_threshold_snr"]))
            if "precheck_radius_px" in params:
                self.precheck_radius_spin.setValue(int(params["precheck_radius_px"]))
            if "precheck_kmad" in params:
                self.precheck_kmad_spin.setValue(float(params["precheck_kmad"]))
            if "fit_enabled" in params:
                self.fit_enabled_check.setChecked(bool(params["fit_enabled"]))
            if "auto_centroid" in params:
                self.auto_centroid_check.setChecked(bool(params["auto_centroid"]))
            if "initial_sigma" in params:
                self.initial_sigma_spin.setValue(float(params["initial_sigma"]))
            if "sigma_min" in params:
                self.sigma_min_spin.setValue(float(params["sigma_min"]))
            if "sigma_max" in params:
                self.sigma_max_spin.setValue(float(params["sigma_max"]))
            if "margin" in params:
                self.margin_spin.setValue(int(params["margin"]))
            if "min_amplitude" in params:
                self.min_amp_spin.setValue(float(params["min_amplitude"]))
            if "min_sigma_result" in params:
                self.min_sigma_result_spin.setValue(float(params["min_sigma_result"]))
            if "snap_enabled" in params:
                self.snap_enabled_check.setChecked(bool(params["snap_enabled"]))
            if "snap_radius" in params:
                self.snap_radius_spin.setValue(int(params["snap_radius"]))
            if "snap_refit_enabled" in params:
                self.snap_refit_check.setChecked(bool(params["snap_refit_enabled"]))
            if "height_export_mode" in params:
                mode = str(params["height_export_mode"]).strip().lower()
                if mode == "point":
                    self.height_export_mode_combo.setCurrentText("Spot Position")
                else:
                    self.height_export_mode_combo.setCurrentText("Spot-Radius Mean")
        finally:
            blockers.clear()

        self._update_detection_ui_enabled()
        self._update_init_mode_ui_enabled()
        self._update_fit_ui_enabled()

    def import_csv_restore(self) -> None:
        """
        Load exported spot CSVs and restore the spot state for display.
        If <base>_meta.json exists, restore ROI / UI params / analysis signatures too.
        """
        if not hasattr(gv, "files") or not gv.files or getattr(gv, "currentFileNum", -1) < 0:
            QtWidgets.QMessageBox.warning(self, "No File", "Load data first.")
            return
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize") or gv.XScanSize == 0 or gv.YScanSize == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Scan Size", "scan_size is 0. Load valid data first.")
            return
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel") or gv.XPixel == 0 or gv.YPixel == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Pixel Size", "Could not get pixel-size information.")
            return

        if self.export_dir is None and hasattr(gv, "files") and gv.files:
            try:
                self.export_dir = os.path.dirname(gv.files[gv.currentFileNum])
            except Exception:
                pass

        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open spot CSV (base or pN)",
            self.export_dir or "",
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return

        export_dir = os.path.dirname(selected_path)
        base_stub = os.path.splitext(os.path.basename(selected_path))[0]
        # allow selecting my_result_p1.csv etc.
        m = re.match(r"^(.*)_p\d+(?:_h)?(?:_\d+)?$", base_stub)
        if m:
            base_stub = m.group(1)
        if not base_stub:
            QtWidgets.QMessageBox.warning(self, "Invalid File", "Could not determine the base name.")
            return

        nm_per_pixel_x = gv.XScanSize / gv.XPixel
        nm_per_pixel_y = gv.YScanSize / gv.YPixel

        restored = self._load_seed_spots_from_csv(export_dir, base_stub, nm_per_pixel_x, nm_per_pixel_y)
        if not restored:
            QtWidgets.QMessageBox.information(self, "No Data", "No readable spot CSV files were found.")
            return

        # restore in-memory state (display)
        self.spots_by_frame = restored
        self.centroid_reference_by_frame = {int(fi): [dict(pk) for pk in spots] for fi, spots in restored.items()}
        self.preserve_existing_spots_on_auto = True
        self.export_btn.setEnabled(True)
        self.export_dir = export_dir

        # load meta.json if exists
        meta_path = os.path.join(export_dir, f"{base_stub}_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.manual_roi = meta.get("manual_roi") or self.manual_roi
                roi_by_frame = meta.get("roi_by_frame")
                if isinstance(roi_by_frame, dict):
                    # json keys might be strings
                    self.roi_by_frame = {int(k): v for k, v in roi_by_frame.items()}
                ui_params = meta.get("ui_params")
                if isinstance(ui_params, dict):
                    self._apply_ui_params(ui_params)
                sig_map = meta.get("analysis_signature_by_frame")
                if isinstance(sig_map, dict):
                    def _to_tuple(obj):
                        if isinstance(obj, list):
                            return tuple(_to_tuple(v) for v in obj)
                        if isinstance(obj, dict):
                            return {k: _to_tuple(v) for k, v in obj.items()}
                        return obj
                    self._analysis_signature_by_frame = {int(k): _to_tuple(v) for k, v in sig_map.items()}
            except Exception:
                pass

        # ensure signatures exist for restored frames to prevent unwanted auto rerun
        for fi in list(self.spots_by_frame.keys()):
            if fi in self._analysis_signature_by_frame:
                continue
            roi_info = self.roi_by_frame.get(fi, self.manual_roi)
            if roi_info is None:
                continue
            try:
                self._analysis_signature_by_frame[int(fi)] = self._analysis_signature(int(fi), roi_info)
            except Exception:
                pass

        # refresh view
        frame = self._prepare_frame()
        if frame is not None:
            self.last_frame = frame
        try:
            self.roi_status_label.setText("ROI Selected" if self.manual_roi is not None else "ROI Not Selected")
        except Exception:
            pass
        self._sync_run_buttons_enabled()
        self.show_full_image_view()
        self.export_btn.setEnabled(True)
        self._refresh_overlay()

    # backward-compatible alias
    def import_csv_and_rerun(self) -> None:
        return self.import_csv_restore()

    def _load_seed_spots_from_csv(
        self, export_dir: str, base_stub: str, nm_per_pixel_x: float, nm_per_pixel_y: float
    ) -> Dict[int, List[Dict[str, float]]]:
        """
        Load <base>_pN.csv and optionally <base>_pN_h.csv.
        Returns frame_index -> list of spot dicts (absolute px).
        The restored list order is guaranteed to follow file labels:
        p1 -> index 0, p2 -> index 1, ...
        """
        # Choose best candidate per pN: prefer no suffix, then lowest suffix
        pos_paths = glob.glob(os.path.join(export_dir, f"{base_stub}_p*.csv"))
        # exclude height files
        pos_paths = [p for p in pos_paths if not re.search(r"_p\d+_h(?:_\d+)?\.csv$", os.path.basename(p))]

        by_p: Dict[int, List[Tuple[int, str]]] = {}
        for path in pos_paths:
            name = os.path.basename(path)
            m = re.match(rf"^{re.escape(base_stub)}_p(\d+)(?:_(\d+))?\.csv$", name)
            if not m:
                continue
            p_idx = int(m.group(1))
            suf = int(m.group(2)) if m.group(2) is not None else 0
            by_p.setdefault(p_idx, []).append((suf, path))
        if not by_p:
            return {}

        best_pos_by_p: Dict[int, str] = {p: sorted(lst, key=lambda t: t[0])[0][1] for p, lst in by_p.items()}

        seed_by_frame_and_p: Dict[int, Dict[int, Dict[str, float]]] = {}
        for p_idx, pos_path in best_pos_by_p.items():
            try:
                with open(pos_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            frame_i = int(float(row.get("frame_index", "nan")))
                            x_nm = float(row.get("x_nm", "nan"))
                            y_nm = float(row.get("y_nm", "nan"))
                        except Exception:
                            continue
                        if not np.isfinite(x_nm) or not np.isfinite(y_nm):
                            continue
                        x_px = x_nm / float(nm_per_pixel_x)
                        y_px = y_nm / float(nm_per_pixel_y)
                        frame_map = seed_by_frame_and_p.setdefault(frame_i, {})
                        frame_map[int(p_idx)] = {
                            "x": float(x_px),
                            "y": float(y_px),
                            "snr": 0.0,
                        }
            except Exception:
                continue

            # merge height if exists (prefer no suffix like position)
            base_name = os.path.splitext(os.path.basename(pos_path))[0]
            # base_name is <base>_pN or <base>_pN_k ; derive height file
            h_candidates = []
            for hp in glob.glob(os.path.join(export_dir, f"{base_name}_h*.csv")):
                h_candidates.append(hp)
            # fallback: any base_stub_pN_h*.csv
            if not h_candidates:
                h_candidates = glob.glob(os.path.join(export_dir, f"{base_stub}_p{p_idx}_h*.csv"))
            h_path = None
            if h_candidates:
                # prefer exact match without extra suffix
                h_candidates.sort(key=lambda p: len(os.path.basename(p)))
                h_path = h_candidates[0]
            if h_path:
                try:
                    with open(h_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                frame_i = int(float(row.get("frame_index", "nan")))
                            except Exception:
                                continue
                            frame_map = seed_by_frame_and_p.get(frame_i)
                            if not frame_map:
                                continue
                            spot = frame_map.get(int(p_idx))
                            if spot is None:
                                continue
                            try:
                                h_value = float(row.get("height_value_nm", "nan"))
                                h_mode = str(row.get("height_mode", "") or "").strip().lower()
                                h_point = float(row.get("height_point_nm", "nan"))
                                h_point_bgsub = float(row.get("height_point_bgsub_nm", "nan"))
                                h_mean = float(row.get("height_mean_nm", "nan"))
                                h_mean_bgsub = float(row.get("height_mean_bgsub_nm", "nan"))
                                h_bg = float(row.get("height_bg_nm", "nan"))
                                h_bgsub = float(row.get("height_bgsub_nm", "nan"))
                            except Exception:
                                continue
                            if not np.isfinite(h_point) and h_mode == "point" and np.isfinite(h_value):
                                h_point = h_value
                            if not np.isfinite(h_point_bgsub) and h_mode == "point" and np.isfinite(h_bgsub):
                                h_point_bgsub = h_bgsub
                            if not np.isfinite(h_mean) and (h_mode == "mean" or not h_mode) and np.isfinite(h_value):
                                h_mean = h_value
                            if not np.isfinite(h_mean_bgsub) and (h_mode == "mean" or not h_mode) and np.isfinite(h_bgsub):
                                h_mean_bgsub = h_bgsub
                            if not np.isfinite(h_mean) and np.isfinite(h_value) and not np.isfinite(h_point):
                                # backward-compatible fallback for legacy files
                                h_mean = h_value
                            if not np.isfinite(h_mean_bgsub) and np.isfinite(h_bgsub) and not np.isfinite(h_point_bgsub):
                                h_mean_bgsub = h_bgsub
                            if not np.isfinite(h_bgsub):
                                if h_mode == "point" and np.isfinite(h_point_bgsub):
                                    h_bgsub = h_point_bgsub
                                elif np.isfinite(h_mean_bgsub):
                                    h_bgsub = h_mean_bgsub
                            spot["height_mean_nm"] = h_mean
                            spot["height_mean_bgsub_nm"] = h_mean_bgsub
                            spot["height_point_nm"] = h_point
                            spot["height_point_bgsub_nm"] = h_point_bgsub
                            spot["height_bg_nm"] = h_bg
                            spot["height_bgsub_nm"] = h_bgsub
                except Exception:
                    pass

        restored: Dict[int, List[Dict[str, float]]] = {}
        for frame_i, frame_map in seed_by_frame_and_p.items():
            ordered = [frame_map[p] for p in sorted(frame_map.keys())]
            if ordered:
                restored[int(frame_i)] = ordered
        return restored
    # --- helper ---
    def _ensure_live_window(self, win, cls):
        """Regenerate the window if the Qt object has been destroyed"""
        if win is not None and self._is_window_live(win):
            return win
        return cls(self)

    def _is_window_live(self, win) -> bool:
        try:
            _ = win.winId()  # RuntimeError if deleted
            return True
        except RuntimeError:
            return False

def create_plugin(main_window) -> QtWidgets.QWidget:
    """
    Factory called from the Plugin menu.
    Receives the main window and returns a SpotAnalysisWindow.
    """
    return SpotAnalysisWindow(main_window)
