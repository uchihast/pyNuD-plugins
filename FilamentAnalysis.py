"""
FilamentAnalysis
----------------
AFM画像上のDNA鎖（紐状構造）の輪郭長を計測するプラグイン。
リッジ検出・経路探索・スプライン補間による弧長積分で高精度に輪郭長を算出する。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate, integrate, ndimage
from skimage import filters as skfilters, morphology, restoration
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse

import globalvals as gv
from fileio import LoadFrame, InitializeAryDataFallback

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Filament Analysis"

# UI label -> matplotlib marker for start/end/waypoint dots
MARKER_SHAPE_MAP = {"丸": "o", "四角": "s", "三角上": "^", "三角下": "v", "菱形": "D"}
MARKER_SHAPE_OPTIONS = [
    ("circle", "丸", "Circle", "o"),
    ("square", "四角", "Square", "s"),
    ("triangle_up", "三角上", "Triangle up", "^"),
    ("triangle_down", "三角下", "Triangle down", "v"),
    ("diamond", "菱形", "Diamond", "D"),
]
ANALYSIS_MODE_DNA = "dna"
ANALYSIS_MODE_BEADS = "beads"
ANALYSIS_MODE_OPTIONS = [
    ("連続紐状 (DNA/リッジ追従)", ANALYSIS_MODE_DNA),
    ("セグメント構造（点指定）", ANALYSIS_MODE_BEADS),
]
ANALYSIS_MODE_SHORT_LABEL = {
    ANALYSIS_MODE_DNA: "連続紐状",
    ANALYSIS_MODE_BEADS: "セグメント構造",
}
FILAMENT_BOX_BACKGROUND_PERCENTILE = 20.0
FILAMENT_BOX_GLOBAL_BASELINE_PERCENTILE = 10.0

UI_LANG_JA = "ja"
UI_LANG_EN = "en"

FILAMENT_UI_TRANSLATIONS = {
    "Language": "言語",
    "JPN / EN": "JPN / EN",
    "Japanese": "日本語",
    "English": "英語",
    "操作内容と出力ファイルの説明を表示します": "Show descriptions of controls and exported files",
    "UI表示を日本語/英語に切り替えます": "Switch the UI language between Japanese and English",
    "ROI / 基本": "ROI / Basics",
    "ROI未選択": "ROI not selected",
    "ROI選択済み": "ROI selected",
    "カラーマップ": "Colormap",
    "AFM画像の表示カラーマップ": "Display colormap for the AFM image",
    "点のサイズ (pt)": "Marker size (pt)",
    "始点・終点・通過点のマーカーサイズ": "Marker size for start, end, and waypoint dots",
    "点の形": "Marker shape",
    "始点・終点・通過点のマーカー形状": "Marker shape for start, end, and waypoint dots",
    "解析モード": "Analysis mode",
    "連続紐状 (DNA/リッジ追従)": "Continuous filament (DNA/ridge tracing)",
    "セグメント構造（点指定）": "Segmented structure (point-specified)",
    "連続紐状": "Continuous filament",
    "セグメント構造": "Segmented structure",
    "連続紐状(DNA)はリッジ追従、セグメント構造は点を直線接続": "Continuous filament follows ridges; segmented mode connects specified points with straight lines",
    "リッジ条件": "Ridge criteria",
    "リッジ σ (px)": "Ridge sigma (px)",
    "リッジ検出に使うガウシアン幅。繊維の見かけ幅に近い値から調整します": "Gaussian width used for ridge detection. Start near the apparent filament width.",
    "最大曲げ角度 (deg)": "Max bending angle (deg)",
    "オフ": "Off",
    "交差点で急角度を禁止。0=オフ。DNAの持続長に合わせて30–90程度を推奨": "Prevents sharp turns at crossings. 0=off. 30-90 is a typical range for DNA persistence.",
    "リッジ重み": "Ridge weight",
    "コストでリッジを重視。高めにするとU字の内側を通りにくい": "Weight for ridge preference in path cost. Higher values reduce U-turn shortcuts.",
    "リッジ閾値（通過禁止）": "Ridge floor threshold",
    "リッジがこの値未満のピクセルを通れなくする。0=オフ。端点付近は除外": "Blocks pixels below this ridge score. 0=off. Endpoint neighborhoods are exempt.",
    "前処理（測長用ROI）": "Preprocessing (ROI for tracing)",
    "ノイズ軽減（メディアン）": "Noise reduction (median)",
    "測長前にメディアンフィルタでノイズを軽減": "Apply a median filter before contour measurement",
    "カーネル k": "Kernel k",
    "0=オフ。奇数推奨（3,5,7）": "0=off. Odd values are recommended (3, 5, 7)",
    "デコンボリューション (Richardson-Lucy)": "Deconvolution (Richardson-Lucy)",
    "探針ボケを軽減し、重なった鎖の境界をシャープに": "Reduces probe blur and sharpens boundaries of overlapping filaments",
    "デコンボ PSF σ (px)": "Deconv PSF sigma (px)",
    "Richardson-LucyデコンボリューションのPSF幅。大きすぎると過補正になります": "PSF width for Richardson-Lucy deconvolution. Too large a value can over-correct the image.",
    "デコンボ 反復回数": "Deconv iterations",
    "デコンボリューションの反復回数。増やすとシャープになりますがノイズも増えます": "Number of deconvolution iterations. More iterations sharpen the image but can amplify noise.",
    "背景処理": "Background handling",
    "なし": "None",
    "平面引き": "Plane subtraction",
    "ガウシアン引き": "Gaussian subtraction",
    "ROI内の傾き・ドリフトを除去": "Remove tilt or drift inside the ROI",
    "背景 σ (px)": "Background sigma (px)",
    "ガウシアン引き時のみ。鎖幅より大きく": "Used only for Gaussian subtraction. Set larger than the filament width.",
    "コントラスト強調": "Contrast enhancement",
    "パーセンタイル": "Percentile",
    "ガンマ": "Gamma",
    "測長前のコントラスト調整": "Adjust contrast before contour measurement",
    "パーセンタイル Low": "Percentile low",
    "パーセンタイル強調時の下限。これ以下を暗側に割り当てます": "Lower bound for percentile contrast. Values below this are mapped to the dark side.",
    "パーセンタイル High": "Percentile high",
    "パーセンタイル強調時の上限。これ以上を明側に割り当てます": "Upper bound for percentile contrast. Values above this are mapped to the bright side.",
    "ガンマ補正値。1で無補正、1未満で暗部を強調します": "Gamma correction value. 1 means no correction; below 1 emphasizes darker features.",
    "輪郭長": "Contour length",
    "ROI画像上で左クリックで始点・終点。U字などは Shift+Click で通過点を追加": "Left-click start and end points on the ROI image. Use Shift+Click to add waypoints for U-shaped filaments.",
    "現在のトレースを記録表に追加し、Full AFM画像へ輪郭を描画します": "Add the current trace to the recorded table and draw the contour on the full AFM image",
    "クリック済み位置が残っていて輪郭が未計算の場合は、現在フレームで輪郭を再計算してから記録します": "If clicked points remain but no contour is calculated, recalculate the contour on the current frame before recording",
    "同じファイル・同じフレーム・同じ繊維グループの既存記録がある場合は、古い記録を上書きします": "If a record already exists for the same file, frame, and fiber group, the old record is overwritten",
    "位置クリア": "Clear Points",
    "クリック済みの始点・終点・通過点を消去します": "Clear clicked start, end, and waypoint positions",
    "OKで現在グループに追加": "Add to current group on OK",
    "ONの場合、OKで記録すると現在の繊維グループIDも保存します": "When ON, OK also saves the current fiber group ID with the record",
    "点数": "Points",
    "指定点数": "Specified points",
    "Persistence Length も計算して記録": "Also calculate and record Persistence Length",
    "ON時のみ Persistence Length を計算し、表示・記録します": "Only when ON, calculate, display, and record Persistence Length",
    "計算待ち": "waiting",
    "計算不可": "not available",
    "スケール不明": "scale unknown",
    "記録した測定値": "Recorded Measurements",
    "記録した測定値表の全行をCSV保存します。初期フォルダは現在のAFMデータフォルダです": "Save all rows in the recorded measurements table as CSV. The initial folder is the current AFM data folder.",
    "モード": "Mode",
    "直線化 / Linearization": "Linearization",
    "記録済み輪郭から全フレームの直線化boxを一括作成し、カタログを開きます": "Create linearized boxes from all recorded contours and open the catalog",
    "中心線の左右に取る半幅です。box全幅はおよそこの値の2倍になります": "Half width sampled on both sides of the centerline. The total box width is approximately twice this value.",
    "Box内1次傾き補正": "First-order box tilt correction",
    "Linearization後の各box内で一次面 z = a*s + b*offset + c をfitして差し引きます。": "After Linearization, fit z = a*s + b*offset + c inside each box and subtract it.",
    "作成済みの直線化boxカタログを開きます。再計算や保存は行いません": "Open the existing linearized box catalog. This does not recalculate or save files.",
    "揺らぎ解析": "Fluctuation Analysis",
    "繊維グループ:": "Fiber group:",
    "未割り当て": "Unassigned",
    "新規グループ開始": "Start New Group",
    "新しい繊維グループIDを作成し、以後のOK記録の割り当て先にします": "Create a new fiber group ID and use it as the assignment target for subsequent OK records",
    "現在グループに追加": "Add to Current Group",
    "現在のトレースをOKと同じ処理で記録し、現在グループに割り当てます": "Record the current trace with the same processing as OK and assign it to the current group",
    "現在グループ削除": "Delete Current Group",
    "現在グループの割り当てを解除します。測定レコードやbox自体は削除しません": "Remove the current group assignment. Measurement records and boxes themselves are not deleted.",
    "グループ情報": "Group Information",
    "フレームレート (fps)": "Frame rate (fps)",
    "frame_indexを秒に変換するフレームレートです。0.5 fpsなら1フレーム2秒です": "Frame rate used to convert frame_index to seconds. At 0.5 fps, one frame is 2 seconds.",
    "リサンプル点数": "Resample points",
    "各輪郭を等弧長にそろえる点数です。大きいほど細かくなりますがノイズも拾いやすくなります": "Number of equal-arc-length samples per contour. Larger values are finer but more sensitive to noise.",
    "揺らぎ解析を実行": "Run Fluctuation Analysis",
    "fiber_group_idが付いた記録をグループごとに曲率ゆらぎ解析します": "Run curvature fluctuation analysis for records with fiber_group_id, grouped by fiber",
    "未実行": "Not run",
    "解析不可": "Analysis unavailable",
    "CSV保存": "Save CSV",
    "曲率行列、分散/Lp_local、自己相関をグループ別CSVで保存します": "Save curvature matrices, variance/Lp_local, and autocorrelation as per-group CSV files",
    "直線化ボックスがありません。": "No linearized boxes are available.",
    "デフォルト保存フォルダを作成できませんでした": "Could not create the default save folder",
    "出力に失敗しました": "Export failed",
    "直線化ボックスを出力しました": "Linearized boxes were exported",
    "この行を削除": "Delete this row",
    "記録した測定値がありません。": "No recorded measurements are available.",
    "CSV保存に失敗しました": "CSV save failed",
    "CSVを保存しました": "CSV was saved",
    "上書き確認": "Overwrite confirmation",
    "上書き": "Overwrite",
    "ファイル名を変更": "Change filename",
    "セッションを保存しました": "Session was saved",
    "保存に失敗しました": "Save failed",
    "読み込みに失敗しました": "Load failed",
    "不明なバージョンです": "Unknown version",
    "セッションを読み込みました。": "Session was loaded.",
    "セッションを読み込みました。同じファイルを開くとオーバーレイが表示されます。": "Session was loaded. Open the same file to display overlays.",
    "輪郭を計測してからOKを押してください。": "Measure a contour before pressing OK.",
    "未記録の輪郭": "Unrecorded contour",
    "OKで記録していない点または輪郭があります。Nextで次フレームへ移動しますか？": "There are points or a contour not recorded with OK. Move to the next frame?",
    "記録する場合はキャンセルしてOKを押してください。": "To record it, cancel and press OK.",
    "移動": "Move",
    "キャンセル": "Cancel",
    "先に新規グループを開始してください。": "Start a new group first.",
    "削除する現在グループがありません。": "There is no current group to delete.",
    "現在グループ削除": "Delete Current Group",
    "測定レコードと直線化box自体は削除しません。": "Measurement records and linearized boxes themselves will not be deleted.",
    "nm/pxスケールを取得できません。": "Could not obtain the nm/px scale.",
    "割り当て済みの繊維グループがありません。": "There are no assigned fiber groups.",
    "解析可能なグループがありません。": "There are no analyzable groups.",
    "フィット不可": "fit unavailable",
    "グループ解析完了": "groups analyzed",
    "直線化できる記録済み輪郭がありません。": "There are no recorded contours available for linearization.",
    "既存の直線化ボックスを置き換えて、記録済み輪郭から再作成しますか？": "Replace existing linearized boxes and recreate them from recorded contours?",
    "ユーザー操作により中断しました。": "Canceled by user.",
    "直線化ボックスを作成できませんでした。": "Could not create linearized boxes.",
    "スキップ": "Skipped",
    "個の直線化ボックスを作成しました。": "linearized boxes were created.",
    "ファイルが見つかりません": "File not found",
    "範囲外です": "is out of range",
    "画像データを読み込めません。": "Could not load image data.",
    "2D画像データを取得できません。": "Could not obtain 2D image data.",
    "FileListが見つかりません。": "FileList was not found.",
    "ファイルリストで対象を選択してください。": "Select a target in the file list.",
    "ファイルがロードされていません。": "No files are loaded.",
    "フレーム読み込みに失敗": "Frame load failed",
    "画像がロードされていません。": "No image is loaded.",
    "Filament Linearization Catalog": "フィラメント直線化カタログ",
    "Export Boxes": "Export Boxes",
    "summary/profile CSV、各boxのstrip CSV/NPY/PNG、軸CSVを保存します。profiles CSVにはs_nmごとの全値が入ります": "Save summary/profile CSV files, each box strip as CSV/NPY/PNG, and axis CSV files. The profiles CSV contains all per-s_nm values.",
    "No linearized boxes": "直線化ボックスがありません",
    "empty": "empty",
    "box half width": "box half width",
    "plane correction": "plane correction",
    "profile points": "profile points",
    "on": "on",
    "off": "off",
    "Filament Fluctuation Analysis": "フィラメント揺らぎ解析",
    "Full AFM image (drag to set ROI)": "Full AFM image (drag to set ROI)",
    "ROI (select region above)": "ROI (select region above)",
    "ROI(DNA): 左クリックで始点・終点。U字などは Shift+Click で通過点を追加": "ROI (DNA): left-click start and end points. Use Shift+Click to add waypoints for U-shaped filaments.",
    "ROI(セグメント): 左クリックで始点・終点、Shift+Clickで中点。点を直線で接続": "ROI (segments): left-click start and end; Shift+Click for intermediate points. Points are connected by straight lines.",
    "前のフレームへ移動します。ROIとクリック位置は残し、計算済み輪郭だけ消去します": "Move to the previous frame. The ROI and clicked positions are kept, and only the calculated contour is cleared.",
    "次のフレームへ移動します。ROIとクリック位置は残し、計算済み輪郭だけ消去します": "Move to the next frame. The ROI and clicked positions are kept, and only the calculated contour is cleared.",
    "記録した測定値と輪郭オーバーレイを消去します。直線化box、Sessionファイル、Export済みファイルは削除しません": "Clear recorded measurements and contour overlays. Linearized boxes, session files, and exported files are not deleted.",
    "記録済み輪郭、グループ、ROI、直線化box、解析設定をJSON保存します": "Save recorded contours, groups, ROI, linearized boxes, and analysis settings to JSON",
    "Session JSONを読み込み、途中までの解析状態を復元します": "Load a session JSON and restore the intermediate analysis state",
}

_FILAMENT_UI_REVERSE = {v: k for k, v in FILAMENT_UI_TRANSLATIONS.items()}

FILAMENT_UI_EN_TO_JA = {
    "Language": "言語",
    "Japanese": "日本語",
    "English": "英語",
    "Help": "ヘルプ",
    "Filament Analysis Help": "Filament Analysisヘルプ",
    "Linearization": "直線化",
    "Catalog": "カタログ",
    "Box half width": "Box半幅",
    "CSV Save": "CSV保存",
    "Data Clear": "データ消去",
    "Clear Points": "位置クリア",
    "Save Session": "セッション保存",
    "Load Session": "セッション読込",
    "Prev": "前へ",
    "Next": "次へ",
    "Frame": "フレーム",
    "Export Boxes": "Box書き出し",
    "No linearized boxes": "直線化ボックスがありません",
    "Filament Linearization Catalog": "フィラメント直線化カタログ",
    "Filament Fluctuation Analysis": "フィラメント揺らぎ解析",
    "Filament Analysis: ROI & Trace": "Filament Analysis: ROIとトレース",
    "Full AFM image (drag to set ROI)": "Full AFM画像（ドラッグでROI指定）",
    "ROI (select region above)": "ROI（上で領域を選択）",
    "empty": "空",
    "box half width": "box半幅",
    "plane correction": "傾き補正",
    "profile points": "profile点数",
    "on": "ON",
    "off": "OFF",
}
_FILAMENT_UI_EN_REVERSE = {v: k for k, v in FILAMENT_UI_EN_TO_JA.items()}


def _ui_text(text: str, lang: str) -> str:
    if lang == UI_LANG_JA and str(text) in FILAMENT_UI_EN_TO_JA:
        return FILAMENT_UI_EN_TO_JA[str(text)]
    if lang == UI_LANG_EN and str(text) in FILAMENT_UI_EN_TO_JA:
        return str(text)
    if lang == UI_LANG_EN and str(text) in _FILAMENT_UI_EN_REVERSE:
        return _FILAMENT_UI_EN_REVERSE[str(text)]
    if lang == UI_LANG_EN:
        return FILAMENT_UI_TRANSLATIONS.get(str(text), str(text))
    return _FILAMENT_UI_REVERSE.get(str(text), str(text))


# ---------------------------------------------------------------------------
# ROI preprocessing for measurement (noise, background, contrast)
# ---------------------------------------------------------------------------

def apply_roi_preprocess(
    roi_image: np.ndarray,
    median_k: int = 0,
    deconv_enable: bool = False,
    deconv_sigma: float = 1.0,
    deconv_iterations: int = 10,
    flatten_mode: str = "none",
    flatten_sigma: float = 10.0,
    contrast_mode: str = "none",
    contrast_low: float = 2.0,
    contrast_high: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Apply optional preprocessing to ROI for contour measurement.
    Order: noise reduction -> deconvolution -> background flattening -> contrast.
    """
    out = np.asarray(roi_image, dtype=np.float64).copy()
    if out.size == 0 or out.ndim != 2:
        return out

    # 1. Noise reduction: median filter
    if median_k and int(median_k) >= 2:
        k = int(median_k)
        if k % 2 == 0:
            k += 1
        try:
            out = skfilters.median(out, footprint=morphology.square(k))
        except Exception:
            pass

    # 2. Deconvolution (Richardson-Lucy): sharpen probe blur
    if deconv_enable and float(deconv_iterations) > 0:
        try:
            v_min, v_max = np.nanmin(out), np.nanmax(out)
            if v_max > v_min:
                img = (out - v_min) / (v_max - v_min)
            else:
                img = np.clip(out, 0, 1).astype(np.float64)
            img = np.maximum(img, 1e-8)
            sigma = max(0.3, float(deconv_sigma))
            size = max(3, int(round(sigma * 4)) * 2 + 1)
            ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
            xx, yy = np.meshgrid(ax, ax)
            psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            psf = psf / np.sum(psf)
            niter = int(deconv_iterations)
            try:
                deconv = restoration.richardson_lucy(img, psf, num_iter=niter)
            except TypeError:
                deconv = restoration.richardson_lucy(img, psf, iterations=niter)
            deconv = np.clip(deconv, 0, None)
            if v_max > v_min:
                out = deconv * (v_max - v_min) + v_min
            else:
                out = deconv.astype(np.float64)
        except Exception:
            pass

    # 3. Background flattening
    flatten_mode = (flatten_mode or "none").strip().lower()
    if flatten_mode == "plane":
        try:
            h, w = out.shape
            yy, xx = np.mgrid[:h, :w]
            X = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
            z = out.ravel()
            ok = np.isfinite(z)
            if np.sum(ok) >= 3:
                coeffs, _, _, _ = np.linalg.lstsq(X[ok], z[ok], rcond=None)
                plane = (X @ coeffs).reshape(h, w)
                out = out - plane
        except Exception:
            pass
    elif flatten_mode == "gaussian" and float(flatten_sigma) > 0:
        try:
            sigma = max(0.5, float(flatten_sigma))
            bg = ndimage.gaussian_filter(out, sigma=sigma, mode="nearest")
            out = out - bg
        except Exception:
            pass

    # 4. Contrast enhancement
    contrast_mode = (contrast_mode or "none").strip().lower()
    if contrast_mode == "percentile":
        try:
            low = max(0, min(100, float(contrast_low)))
            high = max(low, min(100, float(contrast_high)))
            v_min = np.nanpercentile(out, low)
            v_max = np.nanpercentile(out, high)
            if v_max > v_min:
                out = np.clip(out, v_min, v_max)
                out = (out - v_min) / (v_max - v_min)
            else:
                out = np.zeros_like(out)
        except Exception:
            pass
    elif contrast_mode == "gamma" and float(gamma) != 1.0:
        try:
            g = float(gamma)
            v_min, v_max = np.nanmin(out), np.nanmax(out)
            if v_max > v_min:
                out = (out - v_min) / (v_max - v_min)
                out = np.power(np.clip(out, 0, 1), g)
            else:
                out = np.zeros_like(out)
        except Exception:
            pass

    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Algorithm: Ridge detection, path finding, spline arc length
# ---------------------------------------------------------------------------

def compute_ridge_map(roi_image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Frangi filter (Hessian-based ridge detection) for filament center line.
    DNA appears as bright (high) line; ridge map is higher on center line.
    """
    img = np.asarray(roi_image, dtype=np.float64)
    if img.size == 0 or img.ndim != 2:
        return np.zeros_like(img)
    sigmas = [max(0.5, float(sigma))]
    try:
        ridge = skfilters.frangi(img, sigmas=sigmas, black_ridges=False)
    except Exception:
        ridge = np.zeros_like(img)
    return np.asarray(ridge, dtype=np.float64)


def _crop_rect(
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


def _angle_deg_between(ay: float, ax: float, by: float, bx: float) -> float:
    """Angle in degrees between vectors (ay,ax) and (by,bx). 0 = same direction."""
    na = np.sqrt(ay * ay + ax * ax) + 1e-12
    nb = np.sqrt(by * by + bx * bx) + 1e-12
    cos_ab = (ay * by + ax * bx) / (na * nb)
    cos_ab = np.clip(cos_ab, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ab)))


def path_dijkstra_ridge(
    cost_image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_bending_angle_deg: Optional[float] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    8-neighbor Dijkstra on cost image (lower = prefer).
    max_bending_angle_deg: if set, penalize sharp turns (DNA persistence); None = off.
    """
    import heapq

    h, w = cost_image.shape
    sy, sx = int(start[0]), int(start[1])
    ey, ex = int(end[0]), int(end[1])
    if not (0 <= sy < h and 0 <= sx < w and 0 <= ey < h and 0 <= ex < w):
        return None

    use_bending = max_bending_angle_deg is not None and max_bending_angle_deg > 0
    bend_penalty = 50.0

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    dist_scale = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

    INF = float("inf")
    dist = np.full((h, w), INF, dtype=np.float64)
    dist[sy, sx] = 0.0
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    prev[(sy, sx)] = None
    heap: List[Tuple[float, int, int]] = [(0.0, sy, sx)]

    while heap:
        d, y, x = heapq.heappop(heap)
        if (y, x) == (ey, ex):
            break
        if d > dist[y, x]:
            continue
        prev_pt = prev.get((y, x))
        for k, (dy, dx) in enumerate(neighbors):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if use_bending and prev_pt is not None:
                py, px = prev_pt
                in_dy, in_dx = y - py, x - px
                out_dy, out_dx = ny - y, nx - x
                ang = _angle_deg_between(float(in_dy), float(in_dx), float(out_dy), float(out_dx))
                if ang > max_bending_angle_deg:
                    continue
                if ang > max_bending_angle_deg * 0.6:
                    bend_extra = bend_penalty * (ang / max_bending_angle_deg)
                else:
                    bend_extra = 0.0
            else:
                bend_extra = 0.0
            c = float(cost_image[ny, nx])
            if not np.isfinite(c):
                c = 1e6
            step = dist_scale[k] * (0.5 + 0.5 * max(0, min(1, c))) + bend_extra
            nd = d + step
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[(ny, nx)] = (y, x)
                heapq.heappush(heap, (nd, ny, nx))

    if not np.isfinite(dist[ey, ex]):
        return None
    path: List[Tuple[int, int]] = []
    cur = (ey, ex)
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def contour_length_spline(path_pixels: List[Tuple[int, int]]) -> float:
    """
    Fit 3rd-order spline to path and compute arc length by integration.
    path_pixels: list of (row, col) = (y, x). We use (x, y) for spline.
    """
    if len(path_pixels) < 2:
        return 0.0
    xs = np.array([p[1] for p in path_pixels], dtype=np.float64)
    ys = np.array([p[0] for p in path_pixels], dtype=np.float64)
    n = len(xs)
    t = np.linspace(0, 1, n)
    try:
        tck, u = interpolate.splprep([xs, ys], s=0.0, k=min(3, n - 1))
    except Exception:
        return float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))
    u_new = np.linspace(0, 1, max(50, n * 2))
    x_new, y_new = interpolate.splev(u_new, tck)

    def speed(u_val: float) -> float:
        der = interpolate.splev(u_val, tck, der=1)
        return float(np.sqrt(der[0] ** 2 + der[1] ** 2))

    length, _ = integrate.quad(speed, 0, 1, limit=100)
    return float(length)


def end_to_end_distance_from_points(points_xy: List[Tuple[float, float]]) -> Optional[float]:
    """Return straight-line (Euclidean) end-to-end distance in pixels."""
    if points_xy is None or len(points_xy) < 2:
        return None
    x0, y0 = points_xy[0]
    x1, y1 = points_xy[-1]
    return float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))


def persistence_length_2d_from_path(path_pixels: List[Tuple[int, int]], resample_step_px: float = 1.0) -> Optional[float]:
    """
    Estimate persistence length from a traced contour on a 2D plane.
    Uses tangent correlation C(s)=<t(0)·t(s)> and 2D WLC relation C(s)=exp(-s/(2Lp)).
    Returns Lp in pixels. If estimation is unstable, returns None.
    """
    if len(path_pixels) < 6:
        return None
    xs = np.array([p[1] for p in path_pixels], dtype=np.float64)
    ys = np.array([p[0] for p in path_pixels], dtype=np.float64)
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    if seg.size == 0 or not np.all(np.isfinite(seg)):
        return None
    total_len = float(np.sum(seg))
    if total_len <= 0:
        return None
    s = np.concatenate(([0.0], np.cumsum(seg)))
    ds = max(0.5, float(resample_step_px))
    s_new = np.arange(0.0, total_len + 1e-9, ds)
    if s_new.size < 8:
        s_new = np.linspace(0.0, total_len, 8)
    x_new = np.interp(s_new, s, xs)
    y_new = np.interp(s_new, s, ys)

    dxy = np.column_stack([np.diff(x_new), np.diff(y_new)])
    norms = np.linalg.norm(dxy, axis=1)
    ok = norms > 1e-9
    if np.sum(ok) < 6:
        return None
    tangents = dxy[ok] / norms[ok, None]
    s_mid = (0.5 * (s_new[:-1] + s_new[1:]))[ok]
    n = tangents.shape[0]
    if n < 6:
        return None

    max_lag = min(n - 1, max(3, n // 2))
    sep_list: List[float] = []
    corr_list: List[float] = []
    for lag in range(1, max_lag + 1):
        if n - lag < 3:
            break
        dots = np.sum(tangents[:-lag] * tangents[lag:], axis=1)
        corr = float(np.mean(dots))
        if not np.isfinite(corr) or corr <= 0:
            continue
        sep = float(np.mean(s_mid[lag:] - s_mid[:-lag]))
        if sep <= 0 or not np.isfinite(sep):
            continue
        sep_list.append(sep)
        corr_list.append(corr)

    if len(corr_list) < 3:
        return None
    seps = np.asarray(sep_list, dtype=np.float64)
    corrs = np.asarray(corr_list, dtype=np.float64)
    fit_mask = (corrs > 0.05) & (corrs < 0.98)
    if np.sum(fit_mask) >= 3:
        seps = seps[fit_mask]
        corrs = corrs[fit_mask]
    if seps.size < 3:
        return None

    y = np.log(corrs)
    A = np.column_stack([seps, np.ones_like(seps)])
    try:
        slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None
    if not np.isfinite(slope) or slope >= 0:
        return None
    lp_px = -1.0 / (2.0 * slope)
    if not np.isfinite(lp_px) or lp_px <= 0:
        return None
    return float(lp_px)


def persistence_length_2d_from_segments(points_xy: List[Tuple[float, float]]) -> Optional[float]:
    """
    Estimate persistence length from discrete segment chain (clicked bead points).
    Uses tangent correlation on segment vectors and 2D WLC relation C(s)=exp(-s/(2Lp)).
    Returns Lp in pixels.
    """
    if len(points_xy) < 3:
        return None
    pts = np.asarray(points_xy, dtype=np.float64)
    bonds = np.diff(pts, axis=0)
    lens = np.linalg.norm(bonds, axis=1)
    ok = lens > 1e-9
    if np.sum(ok) < 2:
        return None
    bonds = bonds[ok]
    lens = lens[ok]
    tangents = bonds / lens[:, None]
    n = tangents.shape[0]
    if n < 2:
        return None

    s_mid = np.cumsum(lens) - 0.5 * lens
    max_lag = min(n - 1, max(3, n // 2))
    sep_list: List[float] = []
    corr_list: List[float] = []
    for lag in range(1, max_lag + 1):
        if n - lag < 2:
            break
        dots = np.sum(tangents[:-lag] * tangents[lag:], axis=1)
        corr = float(np.mean(dots))
        if not np.isfinite(corr) or corr <= 0:
            continue
        sep = float(np.mean(s_mid[lag:] - s_mid[:-lag]))
        if sep <= 0 or not np.isfinite(sep):
            continue
        sep_list.append(sep)
        corr_list.append(corr)

    if len(corr_list) < 2:
        # Fallback: nearest-neighbor angle only
        dots_nn = np.sum(tangents[:-1] * tangents[1:], axis=1)
        mean_cos = float(np.mean(dots_nn)) if dots_nn.size > 0 else np.nan
        mean_sep = float(np.mean((lens[:-1] + lens[1:]) * 0.5)) if lens.size >= 2 else np.nan
        if not np.isfinite(mean_cos) or mean_cos <= 0 or mean_cos >= 0.999999:
            return None
        if not np.isfinite(mean_sep) or mean_sep <= 0:
            return None
        lp_px = -mean_sep / (2.0 * np.log(mean_cos))
        return float(lp_px) if np.isfinite(lp_px) and lp_px > 0 else None

    seps = np.asarray(sep_list, dtype=np.float64)
    corrs = np.asarray(corr_list, dtype=np.float64)
    fit_mask = (corrs > 0.05) & (corrs < 0.98)
    if np.sum(fit_mask) >= 2:
        seps = seps[fit_mask]
        corrs = corrs[fit_mask]
    if seps.size < 2:
        return None
    y = np.log(corrs)
    A = np.column_stack([seps, np.ones_like(seps)])
    try:
        slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None
    if not np.isfinite(slope) or slope >= 0:
        return None
    lp_px = -1.0 / (2.0 * slope)
    if not np.isfinite(lp_px) or lp_px <= 0:
        return None
    return float(lp_px)


def _path_arc_length_axis(path_xy: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(path_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    finite_rows = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite_rows]
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-12))
    pts = pts[keep]
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.size == 0 or not np.all(np.isfinite(seg)):
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total_len = float(s[-1])
    if total_len <= 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    return pts, s, total_len


def resample_path_uniform(path_xy: List[Tuple[float, float]], n_points: int) -> Tuple[np.ndarray, float]:
    """
    Resample an (x,y) path to equally spaced arc-length points.
    Returns (resampled_xy, total_length) in the same units as the input.
    """
    n = max(2, int(n_points))
    pts, s, total_len = _path_arc_length_axis(path_xy)
    if pts.shape[0] < 2 or s.size < 2:
        return np.empty((0, 2), dtype=np.float64), 0.0
    s_new = np.linspace(0.0, total_len, n)
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x_new, y_new]), total_len


def compute_curvature_from_path(
    xy_resampled: np.ndarray,
    ds: float,
    trim_frac: float = 0.1,
) -> np.ndarray:
    """
    Compute local curvature kappa from an equally arc-length-resampled (x,y) path.
    End regions are replaced with NaN to suppress endpoint derivative artifacts.
    """
    xy = np.asarray(xy_resampled, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return np.empty(0, dtype=np.float64)
    n = xy.shape[0]
    kappa = np.full(n, np.nan, dtype=np.float64)
    ds_val = float(ds)
    if n < 5 or not np.isfinite(ds_val) or ds_val <= 0:
        return kappa
    tangents = (xy[2:] - xy[:-2]) / (2.0 * ds_val)
    theta = np.arctan2(tangents[:, 1], tangents[:, 0])
    theta = np.unwrap(theta)
    if theta.size >= 3:
        kappa[2:-2] = (theta[2:] - theta[:-2]) / (2.0 * ds_val)
    trim_n = int(round(n * max(0.0, min(0.45, float(trim_frac)))))
    if trim_n > 0:
        kappa[:trim_n] = np.nan
        kappa[n - trim_n:] = np.nan
    return kappa


def _finite_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size > 0 else np.nan


def _finite_std(values: np.ndarray, ddof: int = 0) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size <= ddof:
        return np.nan
    return float(np.std(vals, ddof=ddof))


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[col] = _finite_mean(arr[:, col])
    return out


def _nanvar_axis0(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for col in range(arr.shape[1]):
        vals = arr[:, col]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[col] = float(np.var(vals))
    return out


def _nanstd_axis0(values: np.ndarray, ddof: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[col] = _finite_std(arr[:, col], ddof=ddof)
    return out


def _local_persistence_length_from_variance(s_nm: np.ndarray, kappa_var: np.ndarray) -> np.ndarray:
    s_vals = np.asarray(s_nm, dtype=np.float64)
    var_vals = np.asarray(kappa_var, dtype=np.float64)
    lp_local = np.full(var_vals.shape, np.nan, dtype=np.float64)
    if s_vals.size < 2 or var_vals.size == 0:
        return lp_local
    diffs = np.diff(s_vals[np.isfinite(s_vals)])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return lp_local
    ds_nm = float(np.median(diffs))
    if not np.isfinite(ds_nm) or ds_nm <= 0:
        return lp_local
    mask = np.isfinite(var_vals) & (var_vals > 0)
    lp_local[mask] = ds_nm / var_vals[mask]
    return lp_local


def _autocorr_from_delta_kappa(
    delta_kappa: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    t_count = delta_kappa.shape[0]
    c_values: List[float] = [_finite_mean(delta_kappa * delta_kappa)]
    dt_values: List[float] = [0.0]
    if t_count < 2:
        return np.asarray(c_values, dtype=np.float64), np.asarray(dt_values, dtype=np.float64), False

    diffs = np.diff(frame_indices)
    uniform = bool(diffs.size > 0 and np.all(diffs == diffs[0]) and diffs[0] > 0)
    max_count = max(1, t_count // 2)
    if uniform:
        step = int(diffs[0])
        for lag in range(1, max_count):
            products = delta_kappa[:-lag] * delta_kappa[lag:]
            c_values.append(_finite_mean(products))
            dt_values.append(float(step * lag) / float(fps))
    else:
        pair_products: Dict[int, List[float]] = {}
        for i in range(t_count - 1):
            for j in range(i + 1, t_count):
                frame_delta = int(frame_indices[j] - frame_indices[i])
                if frame_delta <= 0:
                    continue
                val = _finite_mean(delta_kappa[i] * delta_kappa[j])
                if np.isfinite(val):
                    pair_products.setdefault(frame_delta, []).append(val)
        for frame_delta in sorted(pair_products.keys())[:max(0, max_count - 1)]:
            vals = np.asarray(pair_products[frame_delta], dtype=np.float64)
            c_values.append(_finite_mean(vals))
            dt_values.append(float(frame_delta) / float(fps))
    return (
        np.asarray(c_values, dtype=np.float64),
        np.asarray(dt_values, dtype=np.float64),
        not uniform,
    )


def _fit_autocorr_exponential(
    dt_s: np.ndarray,
    autocorr: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    dt = np.asarray(dt_s, dtype=np.float64)
    y = np.asarray(autocorr, dtype=np.float64)
    mask = np.isfinite(dt) & np.isfinite(y) & (y > 0)
    if np.sum(mask) < 3 or np.nanmax(dt[mask]) <= 0:
        return None, None, None
    x_fit = dt[mask]
    y_fit = y[mask]

    def model(x, amp, tau):
        return amp * np.exp(-x / tau)

    amp0 = float(y_fit[0]) if np.isfinite(y_fit[0]) and y_fit[0] > 0 else float(np.nanmax(y_fit))
    tau0 = max(float(np.nanmax(x_fit) - np.nanmin(x_fit)), 1e-9)
    try:
        from scipy import optimize

        popt, _pcov = optimize.curve_fit(
            model,
            x_fit,
            y_fit,
            p0=(amp0, tau0),
            bounds=([0.0, 1e-12], [np.inf, np.inf]),
            maxfev=10000,
        )
        amp = float(popt[0])
        tau = float(popt[1])
    except Exception:
        positive_mask = (x_fit > 0) & (y_fit > 0)
        if np.sum(positive_mask) < 2:
            return None, None, None
        A = np.column_stack([x_fit[positive_mask], np.ones(np.sum(positive_mask))])
        try:
            slope, intercept = np.linalg.lstsq(A, np.log(y_fit[positive_mask]), rcond=None)[0]
        except Exception:
            return None, None, None
        if not np.isfinite(slope) or slope >= 0:
            return None, None, None
        tau = float(-1.0 / slope)
        amp = float(np.exp(intercept))
    if not np.isfinite(amp) or not np.isfinite(tau) or tau <= 0:
        return None, None, None
    return amp, tau, model(dt, amp, tau)


def fluctuation_analysis(
    records: List[dict],
    nm_per_px: float,
    fps: float,
    n_resample: int = 100,
    trim_frac: float = 0.1,
) -> dict:
    """
    Quantify local curvature fluctuations for one fiber group across frames.
    Input records must contain path_full_xy and frame_index.
    """
    nm_scale = float(nm_per_px)
    fps_val = float(fps)
    n_points = int(n_resample)
    if not np.isfinite(nm_scale) or nm_scale <= 0:
        raise ValueError("nm_per_px must be positive.")
    if not np.isfinite(fps_val) or fps_val <= 0:
        raise ValueError("fps must be positive.")
    if n_points < 20:
        raise ValueError("n_resample must be at least 20.")

    def _frame_index(rec: dict) -> int:
        try:
            return int(rec.get("frame_index", 0))
        except Exception:
            return 0

    records_sorted = sorted(records, key=_frame_index)
    path_infos: List[Tuple[dict, np.ndarray, np.ndarray, float]] = []
    for rec in records_sorted:
        raw_path = rec.get("path_full_xy") or []
        pts_px = np.asarray(raw_path, dtype=np.float64)
        if pts_px.ndim != 2 or pts_px.shape[1] != 2 or pts_px.shape[0] < 2:
            continue
        pts_nm = pts_px * nm_scale
        pts, s, total_len = _path_arc_length_axis([(float(x), float(y)) for x, y in pts_nm])
        if pts.shape[0] >= 2 and s.size >= 2 and total_len > 0:
            path_infos.append((rec, pts, s, total_len))
    if len(path_infos) < 2:
        raise ValueError("同一グループに有効な輪郭が2フレーム以上必要です。")

    l_min_nm = float(min(info[3] for info in path_infos))
    if not np.isfinite(l_min_nm) or l_min_nm <= 0:
        raise ValueError("有効な輪郭長を計算できません。")
    s_nm = np.linspace(0.0, l_min_nm, n_points)
    ds_nm = float(s_nm[1] - s_nm[0]) if n_points > 1 else np.nan
    if not np.isfinite(ds_nm) or ds_nm <= 0:
        raise ValueError("リサンプル間隔を計算できません。")

    xy_all: List[np.ndarray] = []
    kappa_rows: List[np.ndarray] = []
    frame_indices: List[int] = []
    for rec, pts, s, _total_len in path_infos:
        x_new = np.interp(s_nm, s, pts[:, 0])
        y_new = np.interp(s_nm, s, pts[:, 1])
        xy_resampled = np.column_stack([x_new, y_new])
        kappa = compute_curvature_from_path(xy_resampled, ds_nm, trim_frac=trim_frac)
        xy_all.append(xy_resampled)
        kappa_rows.append(kappa)
        frame_indices.append(_frame_index(rec))

    kappa_all = np.asarray(kappa_rows, dtype=np.float64)
    xy_all_arr = np.asarray(xy_all, dtype=np.float64)
    valid_columns = np.any(np.isfinite(kappa_all), axis=0)
    valid_count = int(np.sum(valid_columns))
    if valid_count < 10:
        raise ValueError("端部カット後の有効な曲率点が10点未満です。")

    kappa_mean = _nanmean_axis0(kappa_all)
    kappa_var = _nanvar_axis0(kappa_all)
    lp_local_nm = _local_persistence_length_from_variance(s_nm, kappa_var)
    delta_kappa = kappa_all - kappa_mean[None, :]
    delta2 = delta_kappa * delta_kappa
    n_valid = np.sum(np.isfinite(delta2), axis=0)
    kappa_var_se = np.full(kappa_var.shape, np.nan, dtype=np.float64)
    se_mask = n_valid > 1
    if np.any(se_mask):
        kappa_var_se[se_mask] = _nanstd_axis0(delta2, ddof=1)[se_mask] / np.sqrt(n_valid[se_mask])

    frame_indices_arr = np.asarray(frame_indices, dtype=int)
    times_s = frame_indices_arr.astype(np.float64) / fps_val
    autocorr, autocorr_dt_s, nonuniform = _autocorr_from_delta_kappa(
        delta_kappa, frame_indices_arr, fps_val
    )
    fit_amp, tau_s, autocorr_fit = _fit_autocorr_exponential(autocorr_dt_s, autocorr)

    mean_xy = np.mean(xy_all_arr, axis=0)
    mean_path_yx = [(float(y), float(x)) for x, y in mean_xy]
    lp_nm = persistence_length_2d_from_path(mean_path_yx, resample_step_px=max(ds_nm, 0.5))
    if lp_nm is not None and (not np.isfinite(lp_nm) or lp_nm <= 0):
        lp_nm = None

    warning = None
    if nonuniform:
        warning = "frame_index が等間隔ではないため、自己相関は実際の Δt ごとに平均しました。"

    return {
        "s_nm": s_nm,
        "kappa_all": kappa_all,
        "kappa_mean": kappa_mean,
        "kappa_var": kappa_var,
        "kappa_var_se": kappa_var_se,
        "lp_local_nm": lp_local_nm,
        "times_s": times_s,
        "frame_indices": frame_indices_arr,
        "autocorr": autocorr,
        "autocorr_dt_s": autocorr_dt_s,
        "autocorr_fit": autocorr_fit,
        "fit_amp": fit_amp,
        "tau_s": tau_s,
        "lp_nm": lp_nm,
        "L_min_nm": l_min_nm,
        "valid_points": valid_count,
        "warning": warning,
    }


def _array_to_json_list(arr: Any) -> list:
    values = np.asarray(arr)
    if values.size == 0:
        return []
    return values.tolist()


def _array_from_payload(payload: Any, ndim: int, dtype=float) -> np.ndarray:
    if payload is None:
        return np.empty((0,) * max(1, ndim), dtype=dtype)
    try:
        arr = np.asarray(payload, dtype=dtype)
    except Exception:
        return np.empty((0,) * max(1, ndim), dtype=dtype)
    if arr.ndim != ndim:
        return np.empty((0,) * max(1, ndim), dtype=dtype)
    return arr


def _filament_finite_values(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _filament_relative_height(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    finite = _filament_finite_values(arr)
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    baseline = float(np.percentile(finite, FILAMENT_BOX_GLOBAL_BASELINE_PERCENTILE))
    return arr - baseline


def _axis_step_nm(axis_nm: np.ndarray, fallback: float = 1.0) -> float:
    values = np.asarray(axis_nm, dtype=np.float64)
    if values.size >= 2:
        diffs = np.abs(np.diff(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return max(float(fallback), 1e-9)


def _sample_bilinear(arr: np.ndarray, y: float, x: float) -> float:
    h, w = arr.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return float("nan")
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]
    return float((1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 + (1 - dx) * dy * v10 + dx * dy * v11)


def _arc_lengths_nm_from_yx(points_yx: np.ndarray, nm_x: float, nm_y: float) -> np.ndarray:
    points = np.asarray(points_yx, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    arc = np.zeros(points.shape[0], dtype=np.float64)
    for i in range(1, points.shape[0]):
        dy = float(points[i, 0] - points[i - 1, 0]) * float(nm_y)
        dx = float(points[i, 1] - points[i - 1, 1]) * float(nm_x)
        arc[i] = arc[i - 1] + math.hypot(dx, dy)
    return arc


def straighten_filament_box(
    frame: np.ndarray,
    points_yx: np.ndarray,
    nm_x: float,
    nm_y: float,
    half_width_nm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a straightened rectangular strip along a traced filament.
    Rows are lateral offsets [nm], columns are arc length s [nm].
    """
    points = np.asarray(points_yx, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if not np.isfinite(nm_x) or not np.isfinite(nm_y) or nm_x <= 0 or nm_y <= 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    rel = _filament_relative_height(frame)
    half_width = max(0.1, float(half_width_nm))
    sample_step_nm = max(0.1, min(float(nm_x), float(nm_y)))
    offsets_nm = np.arange(-half_width, half_width + sample_step_nm * 0.5, sample_step_nm, dtype=np.float64)
    s_nm = _arc_lengths_nm_from_yx(points, nm_x, nm_y)
    strip = np.full((offsets_nm.size, points.shape[0]), np.nan, dtype=np.float64)

    xs_nm = points[:, 1] * float(nm_x)
    ys_nm = points[:, 0] * float(nm_y)
    for i in range(points.shape[0]):
        if i == 0:
            dx = xs_nm[1] - xs_nm[0]
            dy = ys_nm[1] - ys_nm[0]
        elif i == points.shape[0] - 1:
            dx = xs_nm[-1] - xs_nm[-2]
            dy = ys_nm[-1] - ys_nm[-2]
        else:
            dx = xs_nm[i + 1] - xs_nm[i - 1]
            dy = ys_nm[i + 1] - ys_nm[i - 1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            continue
        tx = dx / norm
        ty = dy / norm
        nx = -ty
        ny = tx
        for j, offset in enumerate(offsets_nm):
            sample_x_nm = xs_nm[i] + offset * nx
            sample_y_nm = ys_nm[i] + offset * ny
            strip[j, i] = _sample_bilinear(rel, sample_y_nm / float(nm_y), sample_x_nm / float(nm_x))
    return strip, s_nm, offsets_nm


def subtract_linear_plane_from_strip(
    strip: np.ndarray,
    s_nm: np.ndarray,
    offsets_nm: np.ndarray,
    fit_percentile: float = 70.0,
) -> np.ndarray:
    """
    Subtract a first-order plane z = a*s + b*offset + c from a straightened strip.
    The fit prefers lower-height pixels so the filament ridge itself has less leverage.
    """
    arr = np.asarray(strip, dtype=np.float64)
    s = np.asarray(s_nm, dtype=np.float64)
    offsets = np.asarray(offsets_nm, dtype=np.float64)
    if arr.ndim != 2 or s.ndim != 1 or offsets.ndim != 1 or arr.shape != (offsets.size, s.size):
        return arr.copy()
    if arr.size == 0:
        return arr.copy()

    ss, oo = np.meshgrid(s, offsets)
    finite = np.isfinite(arr) & np.isfinite(ss) & np.isfinite(oo)
    if np.sum(finite) < 3:
        return arr.copy()

    fit_mask = finite
    try:
        cutoff = float(np.nanpercentile(arr[finite], float(fit_percentile)))
        candidate = finite & (arr <= cutoff)
        if np.sum(candidate) >= 3:
            fit_mask = candidate
    except Exception:
        fit_mask = finite

    s0 = float(np.nanmean(ss[fit_mask]))
    o0 = float(np.nanmean(oo[fit_mask]))
    A = np.column_stack([
        ss[fit_mask] - s0,
        oo[fit_mask] - o0,
        np.ones(int(np.sum(fit_mask)), dtype=np.float64),
    ])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, arr[fit_mask], rcond=None)
    except Exception:
        return arr.copy()
    plane = coeffs[0] * (ss - s0) + coeffs[1] * (oo - o0) + coeffs[2]
    corrected = arr - plane
    corrected[~finite] = np.nan
    return corrected


def _crossing_offset(offsets: np.ndarray, values: np.ndarray, start: int, stop: int, half_height: float) -> Optional[float]:
    step = -1 if stop < start else 1
    prev_idx = start
    idx = start + step
    while (idx >= stop if step < 0 else idx <= stop):
        v0 = values[prev_idx]
        v1 = values[idx]
        if np.isfinite(v0) and np.isfinite(v1) and (v0 - half_height) * (v1 - half_height) <= 0:
            if abs(v1 - v0) <= 1e-12:
                return float(offsets[idx])
            frac = (half_height - v0) / (v1 - v0)
            return float(offsets[prev_idx] + frac * (offsets[idx] - offsets[prev_idx]))
        prev_idx = idx
        idx += step
    return None


def compute_filament_box_profiles(strip: np.ndarray, offsets_nm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute local background, peak height above background, peak offset, and FWHM for each strip column.
    """
    arr = np.asarray(strip, dtype=np.float64)
    offsets = np.asarray(offsets_nm, dtype=np.float64)
    if arr.ndim != 2 or offsets.ndim != 1 or arr.shape[0] != offsets.size:
        empty = np.empty(0, dtype=np.float64)
        return {
            "background_nm": empty,
            "height_nm": empty,
            "fwhm_nm": empty,
            "peak_offset_nm": empty,
        }
    n_cols = arr.shape[1]
    background = np.full(n_cols, np.nan, dtype=np.float64)
    height = np.full(n_cols, np.nan, dtype=np.float64)
    fwhm = np.full(n_cols, np.nan, dtype=np.float64)
    peak_offset = np.full(n_cols, np.nan, dtype=np.float64)
    for col_idx in range(n_cols):
        col = arr[:, col_idx]
        finite = np.isfinite(col)
        if np.sum(finite) < 5:
            continue
        values = col.copy()
        try:
            bg = float(np.nanpercentile(values, FILAMENT_BOX_BACKGROUND_PERCENTILE))
        except Exception:
            continue
        corrected = values - bg
        corrected[~np.isfinite(corrected)] = np.nan
        if not np.any(np.isfinite(corrected)):
            continue
        peak_idx = int(np.nanargmax(corrected))
        peak = float(corrected[peak_idx])
        background[col_idx] = bg
        peak_offset[col_idx] = float(offsets[peak_idx])
        if not np.isfinite(peak) or peak <= 0:
            continue
        height[col_idx] = peak
        half = 0.5 * peak
        left = _crossing_offset(offsets, corrected, peak_idx, 0, half)
        right = _crossing_offset(offsets, corrected, peak_idx, offsets.size - 1, half)
        if left is not None and right is not None and right > left:
            fwhm[col_idx] = float(right - left)
    return {
        "background_nm": background,
        "height_nm": height,
        "fwhm_nm": fwhm,
        "peak_offset_nm": peak_offset,
    }


def _strip_to_rgb(strip: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(strip, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    norm = np.clip((arr - float(vmin)) / span, 0.0, 1.0)
    norm[~np.isfinite(norm)] = 0.0
    rgb = np.empty((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(2.4 * norm, 0.0, 1.0)
    rgb[..., 1] = np.clip(2.4 * norm - 0.6, 0.0, 1.0)
    rgb[..., 2] = np.clip(3.5 * norm - 2.5, 0.0, 1.0)
    return np.ascontiguousarray(np.round(rgb * 255.0).astype(np.uint8))


def _rgb_to_pixmap(rgb: np.ndarray) -> QtGui.QPixmap:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    bytes_per_line = 3 * width
    image = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(image.copy())


def _profile_to_pixmap(
    s_nm: np.ndarray,
    height_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    width_px: int,
    height_px: int = 46,
) -> QtGui.QPixmap:
    width = max(1, int(width_px))
    height = max(24, int(height_px))
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(pixmap)
    try:
        painter.fillRect(0, 0, width, height, QtGui.QColor(255, 255, 255))
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 1))
        painter.drawLine(0, height - 5, width, height - 5)

        def _draw(values: np.ndarray, color: QtGui.QColor) -> None:
            vals = np.asarray(values, dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size < 2:
                return
            vmax = float(np.nanpercentile(finite, 95.0))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= 0:
                return
            s = np.asarray(s_nm, dtype=np.float64)
            if s.size == vals.size and s.size >= 2 and np.nanmax(s) > np.nanmin(s):
                xs = (s - float(np.nanmin(s))) / (float(np.nanmax(s)) - float(np.nanmin(s))) * (width - 1)
            else:
                xs = np.linspace(0, width - 1, vals.size)
            ys = (height - 6) - np.clip(vals / vmax, 0.0, 1.0) * (height - 12)
            painter.setPen(QtGui.QPen(color, 1.3))
            prev = None
            for x_val, y_val, v_val in zip(xs, ys, vals):
                if not np.isfinite(v_val) or not np.isfinite(x_val) or not np.isfinite(y_val):
                    prev = None
                    continue
                point = QtCore.QPointF(float(x_val), float(y_val))
                if prev is not None:
                    painter.drawLine(prev, point)
                prev = point

        _draw(height_nm, QtGui.QColor(30, 110, 220))
        _draw(fwhm_nm, QtGui.QColor(230, 130, 25))
    finally:
        painter.end()
    return pixmap


def polyline_length_from_points(points_xy: List[Tuple[float, float]]) -> float:
    """Length of piecewise linear chain through clicked (x,y) points."""
    if len(points_xy) < 2:
        return 0.0
    pts = np.asarray(points_xy, dtype=np.float64)
    dxy = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)))


def polyline_pixels_from_points(
    points_xy: List[Tuple[float, float]],
    image_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Rasterized polyline path as (row, col) pixels from clicked (x,y) points."""
    h, w = image_shape
    if h <= 0 or w <= 0 or len(points_xy) < 2:
        return []
    out: List[Tuple[int, int]] = []
    for i in range(len(points_xy) - 1):
        x0, y0 = points_xy[i]
        x1, y1 = points_xy[i + 1]
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        steps = max(1, int(np.ceil(seg_len * 2.0)))
        xs = np.linspace(x0, x1, steps + 1)
        ys = np.linspace(y0, y1, steps + 1)
        for x, y in zip(xs, ys):
            col = int(round(np.clip(x, 0, w - 1)))
            row = int(round(np.clip(y, 0, h - 1)))
            p = (row, col)
            if not out or out[-1] != p:
                out.append(p)
    return out


def compute_polyline_between_points(
    roi_image: np.ndarray,
    points_xy: List[Tuple[float, float]],
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    Connect clicked points directly with straight segments.
    Returns rasterized path pixels and total polyline length in px.
    """
    if roi_image is None or roi_image.size == 0 or len(points_xy) < 2:
        return None, 0.0
    path = polyline_pixels_from_points(points_xy, roi_image.shape)
    if len(path) < 2:
        return None, 0.0
    return path, polyline_length_from_points(points_xy)


def _ridge_intensity_maps(
    roi_image: np.ndarray,
    ridge_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ridge = compute_ridge_map(roi_image, sigma=ridge_sigma)
    im = np.asarray(roi_image, dtype=np.float64)
    im_min, im_max = np.nanmin(im), np.nanmax(im)
    if im_max > im_min:
        intensity = (im - im_min) / (im_max - im_min)
    else:
        intensity = np.ones_like(im)
    r_min, r_max = np.nanmin(ridge), np.nanmax(ridge)
    if r_max > r_min:
        ridge_n = (ridge - r_min) / (r_max - r_min)
    else:
        ridge_n = np.zeros_like(ridge)
    return ridge_n, intensity


def snap_points_to_filament_centers(
    roi_image: np.ndarray,
    points_xy: List[Tuple[float, float]],
    ridge_sigma: float = 1.5,
    ridge_weight: float = 0.9,
    search_radius_px: float = 6.0,
) -> List[Tuple[float, float]]:
    """
    Treat clicked anchor points as initial guesses and snap each to the local
    maximum of ridge/intensity score in a small neighborhood.
    """
    if roi_image is None or roi_image.size == 0 or not points_xy:
        return points_xy
    h, w = roi_image.shape
    ridge_n, intensity = _ridge_intensity_maps(roi_image, ridge_sigma=ridge_sigma)
    rw = max(0.0, min(1.0, float(ridge_weight)))
    score = rw * ridge_n + (1.0 - rw) * intensity
    radius = max(1, int(round(float(search_radius_px))))
    snapped: List[Tuple[float, float]] = []
    for x, y in points_xy:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(h - 1, cy + radius)
        local = score[y0:y1 + 1, x0:x1 + 1].copy()
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dist2 = (xx - float(x)) ** 2 + (yy - float(y)) ** 2
        local[dist2 > float(radius) ** 2] = np.nan
        local = local - 1e-6 * dist2
        if not np.any(np.isfinite(local)):
            snapped.append((float(cx), float(cy)))
            continue
        best_flat = int(np.nanargmax(local))
        by_local, bx_local = np.unravel_index(best_flat, local.shape)
        snapped.append((float(x0 + bx_local), float(y0 + by_local)))
    return snapped


def compute_contour_between_points(
    roi_image: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    ridge_sigma: float = 1.5,
    ridge_weight: float = 0.9,
    ridge_floor_threshold: float = 0.0,
    max_bending_angle_deg: Optional[float] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    start_xy, end_xy: (x, y) in ROI image coordinates (col, row).
    ridge_weight: 0–1, weight for ridge in cost (higher = prefer ridge, avoid shortcut).
    ridge_floor_threshold: pixels with ridge_n below this get high cost (0 = off).
    Returns (path_pixels as list of (row,col), contour_length_px).
    If path fails, returns (None, 0.0).
    """
    h, w = roi_image.shape
    ridge_n, intensity = _ridge_intensity_maps(roi_image, ridge_sigma=ridge_sigma)
    # Cost: prefer bright + ridge. Higher ridge_weight avoids shortcut through low-ridge interior.
    rw = max(0.0, min(1.0, float(ridge_weight)))
    cost = 1.0 - (rw * ridge_n + (1.0 - rw) * intensity)
    cost = np.clip(cost, 1e-6, 1.0)

    sy, sx = int(round(start_xy[1])), int(round(start_xy[0]))
    ey, ex = int(round(end_xy[1])), int(round(end_xy[0]))
    sy = max(0, min(h - 1, sy))
    sx = max(0, min(w - 1, sx))
    ey = max(0, min(h - 1, ey))
    ex = max(0, min(w - 1, ex))

    # Optional: forbid path through low-ridge pixels (except near start/end for click tolerance)
    if ridge_floor_threshold > 0:
        yy, xx = np.ogrid[:h, :w]
        d_start = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
        d_end = np.sqrt((yy - ey) ** 2 + (xx - ex) ** 2)
        exempt = (d_start <= 3) | (d_end <= 3)
        low_ridge = ridge_n < ridge_floor_threshold
        cost = np.where(low_ridge & ~exempt, 10.0, cost)

    path = path_dijkstra_ridge(
        cost, (sy, sx), (ey, ex), max_bending_angle_deg=max_bending_angle_deg
    )
    if path is None or len(path) < 2:
        return None, 0.0
    length_px = contour_length_spline(path)
    return path, length_px


# ---------------------------------------------------------------------------
# Full image window: top = AFM + ROI selector, bottom = ROI crop
# ---------------------------------------------------------------------------

class ContourLengthFullImageWindow(QtWidgets.QMainWindow):
    """全画像でROI選択、下段にROIクロップを1枚表示。ROI上で2点クリックで端点指定→輪郭長計測。"""

    def __init__(
        self,
        parent=None,
        roi_basic_group: Optional[QtWidgets.QWidget] = None,
        contour_length_group: Optional[QtWidgets.QWidget] = None,
        frame_nav_widget: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._ui_language = getattr(parent, "_ui_language", UI_LANG_JA)
        self.setWindowTitle("Filament Analysis: ROI & Trace")
        self.resize(700, 600)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        h_layout = QtWidgets.QHBoxLayout(central_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        h_layout.addWidget(self.main_splitter)

        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.figure)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.canvas)
        self.main_splitter.addWidget(left_panel)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.3)
        self.ax_afm = self.figure.add_subplot(gs[0])
        self.ax_roi = self.figure.add_subplot(gs[1])

        self.cbar = None
        self.selector = None
        self.on_select_callback = None
        self._endpoints: List[Tuple[float, float]] = []
        self._endpoint_click_count = 0  # 0=none, 1=start set, 2=end set (normal left-clicks only)
        self._shift_held = False
        self._path_overlay = None
        self._path_length: Optional[float] = None
        self._path_pixels: Optional[List[Tuple[int, int]]] = None
        self._ridge_sigma = 1.5
        self._ridge_weight = 0.9
        self._ridge_floor_threshold = 0.0
        self._max_bending_angle_deg: Optional[float] = None
        self._analysis_mode = ANALYSIS_MODE_DNA
        self._anchor_point_count: Optional[int] = None
        self._anchor_points_for_trace: Optional[List[Tuple[float, float]]] = None
        self._snapped_anchor_points_for_trace: Optional[List[Tuple[float, float]]] = None
        self._persistence_path_pixels: Optional[List[Tuple[int, int]]] = None
        self._on_length_computed_callback = None
        self._nm_per_pixel: Optional[float] = None
        self._marker_size = 2
        self._marker_shape = "o"  # matplotlib marker: o, s, ^, v, D, etc.
        self._cmap_name = "viridis"

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.canvas.mpl_connect("key_release_event", self._on_key_release)

        if roi_basic_group is not None:
            roi_basic_group.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            wrapper = QtWidgets.QWidget()
            wrap_layout = QtWidgets.QVBoxLayout(wrapper)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.setAlignment(QtCore.Qt.AlignTop)
            wrap_layout.addWidget(roi_basic_group)
            if contour_length_group is not None:
                contour_length_group.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
                )
                wrap_layout.addWidget(contour_length_group)
            if frame_nav_widget is not None:
                frame_nav_widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
                )
                wrap_layout.addWidget(frame_nav_widget)
            right_scroll = QtWidgets.QScrollArea()
            right_scroll.setWidget(wrapper)
            right_scroll.setWidgetResizable(True)
            right_scroll.setMinimumWidth(180)
            right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.main_splitter.addWidget(right_scroll)
            self.main_splitter.setStretchFactor(0, 1)
            self.main_splitter.setStretchFactor(1, 0)
            self.main_splitter.setSizes([900, 260])

    def _tr(self, text: str) -> str:
        return _ui_text(text, self._ui_language)

    def set_ui_language(self, lang: str) -> None:
        self._ui_language = UI_LANG_EN if lang == UI_LANG_EN else UI_LANG_JA
        self.setWindowTitle(self._tr("Filament Analysis: ROI & Trace"))
        self._draw_roi_with_endpoints_and_path()

    def set_ridge_sigma(self, sigma: float) -> None:
        self._ridge_sigma = max(0.5, float(sigma))

    def set_ridge_weight(self, w: float) -> None:
        """0–1: weight for ridge in cost (higher = prefer ridge, avoid U-shortcut)."""
        self._ridge_weight = max(0.0, min(1.0, float(w)))

    def set_ridge_floor_threshold(self, t: float) -> None:
        """0 = off; pixels with ridge_n below this get high cost (except near start/end)."""
        self._ridge_floor_threshold = max(0.0, float(t))

    def set_max_bending_angle_deg(self, deg: Optional[float]) -> None:
        """None or 0 = off; otherwise limit bending angle (DNA persistence)."""
        if deg is None or float(deg) <= 0:
            self._max_bending_angle_deg = None
        else:
            self._max_bending_angle_deg = max(1.0, float(deg))

    def set_analysis_mode(self, mode: str) -> None:
        if mode == ANALYSIS_MODE_BEADS:
            self._analysis_mode = ANALYSIS_MODE_BEADS
        else:
            self._analysis_mode = ANALYSIS_MODE_DNA

    def set_marker_size(self, size: float) -> None:
        """Marker size in points for start/end/waypoint dots."""
        self._marker_size = max(2.0, min(60.0, float(size)))

    def set_marker_shape(self, mpl_marker: str) -> None:
        """Matplotlib marker character: o, s, ^, v, D, etc."""
        self._marker_shape = str(mpl_marker) if mpl_marker else "o"

    def set_cmap(self, cmap_name: str) -> None:
        """Set colormap for AFM and ROI images. Falls back to 'viridis' if invalid."""
        if not cmap_name or not cmap_name.strip():
            self._cmap_name = "viridis"
            return
        try:
            import matplotlib.pyplot as plt
            plt.get_cmap(cmap_name.strip())
            self._cmap_name = cmap_name.strip()
        except Exception:
            self._cmap_name = "viridis"

    def set_on_length_computed(self, callback) -> None:
        """Main window sets this so that when 2nd endpoint is set we can report length."""
        self._on_length_computed_callback = callback

    def set_nm_per_pixel(self, nm_per_pixel: Optional[float]) -> None:
        """Set scale (nm/px) for displaying contour length in nm."""
        self._nm_per_pixel = float(nm_per_pixel) if nm_per_pixel is not None and nm_per_pixel > 0 else None

    def enable_roi_selector(self, shape: str, callback) -> None:
        from matplotlib.widgets import RectangleSelector, EllipseSelector
        self.on_select_callback = callback
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        def _emit_rect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            if callback:
                callback({"shape": "Rectangle", "x0": xmin, "y0": ymin, "w": xmax - xmin, "h": ymax - ymin})

        def _emit_ellipse(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            rx, ry = (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
            if callback:
                callback({"shape": "Ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "x0": xmin, "y0": ymin, "w": xmax - xmin, "h": ymax - ymin})

        if shape == "Ellipse (Circle)":
            self.selector = EllipseSelector(
                self.ax_afm, _emit_ellipse, useblit=True, button=[1],
                minspanx=2, minspany=2, interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )
        else:
            self.selector = RectangleSelector(
                self.ax_afm, _emit_rect, useblit=True, button=[1],
                minspanx=2, minspany=2, interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )

    def _on_key_press(self, event) -> None:
        if getattr(event, "key", None) == "shift":
            self._shift_held = True

    def _on_key_release(self, event) -> None:
        if getattr(event, "key", None) == "shift":
            self._shift_held = False

    def clear_contour(self) -> None:
        """Clear endpoints, path overlay, and length. Redraw ROI."""
        self._endpoints = []
        self._endpoint_click_count = 0
        self._path_overlay = None
        self._path_length = None
        self._path_pixels = None
        self._anchor_points_for_trace = None
        self._snapped_anchor_points_for_trace = None
        self._persistence_path_pixels = None
        self._anchor_point_count = None
        self._draw_roi_with_endpoints_and_path()

    def clear_trace_result_keep_points(self) -> None:
        """Clear computed path/length while keeping manually clicked points."""
        self._path_overlay = None
        self._path_length = None
        self._path_pixels = None
        self._anchor_points_for_trace = None
        self._snapped_anchor_points_for_trace = None
        self._persistence_path_pixels = None
        self._anchor_point_count = len(self._endpoints) if self._endpoints else None
        self._draw_roi_with_endpoints_and_path()

    def _roi_instruction_text(self) -> str:
        if self._analysis_mode == ANALYSIS_MODE_BEADS:
            return self._tr("ROI(セグメント): 左クリックで始点・終点、Shift+Clickで中点。点を直線で接続")
        return self._tr("ROI(DNA): 左クリックで始点・終点。U字などは Shift+Click で通過点を追加")

    def _on_press(self, event):
        if event.inaxes != self.ax_roi or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        pt = (float(x), float(y))
        # Shift: prefer guiEvent modifier (reliable when clicking); fallback to key_press state
        shift_pressed = self._shift_held
        if hasattr(event, "guiEvent") and event.guiEvent is not None:
            try:
                mod = event.guiEvent.modifiers()
                shift_pressed = bool(mod & QtCore.Qt.ShiftModifier)
            except Exception:
                pass
        if shift_pressed:
            # Shift+Click = waypoint (only when start set, end not set)
            if self._endpoint_click_count == 1:
                self._endpoints.append(pt)
        else:
            # Left click = endpoint: 1st = start, 2nd = end; or "start over" if already 2
            if self._endpoint_click_count == 0:
                self._endpoints.append(pt)
                self._endpoint_click_count = 1
            elif self._endpoint_click_count == 1:
                self._endpoints.append(pt)
                self._endpoint_click_count = 2
            else:
                # Already have start and end: this click = new start (clear and re-specify)
                self._endpoints = [pt]
                self._endpoint_click_count = 1
        self._draw_roi_with_endpoints_and_path()
        if self._endpoint_click_count >= 2 and len(self._endpoints) >= 2 and hasattr(self, "_roi_image") and self._roi_image is not None:
            self.run_trace_and_length(self._roi_image, self._on_length_computed_callback)

    def _draw_roi_with_endpoints_and_path(self) -> None:
        if not hasattr(self, "_roi_image") or self._roi_image is None:
            return
        self.ax_roi.clear()
        self.ax_roi.imshow(self._roi_image, cmap=self._cmap_name, origin="lower")
        n_pt = len(self._endpoints)
        for i, (px, py) in enumerate(self._endpoints):
            if i == 0:
                color = "red"  # start
            elif i == n_pt - 1:
                color = "cyan"  # end
            else:
                color = "yellow"  # waypoint
            self.ax_roi.plot(
                px, py, self._marker_shape, color=color,
                markersize=self._marker_size, markeredgewidth=2,
            )
        snapped = self._snapped_anchor_points_for_trace or []
        if snapped and len(snapped) == n_pt:
            for sx, sy in snapped:
                self.ax_roi.plot(
                    sx, sy, "+", color="white",
                    markersize=max(5.0, self._marker_size + 3.0),
                    markeredgewidth=1.4,
                )
        if self._path_pixels and len(self._path_pixels) >= 2:
            xs = [p[1] for p in self._path_pixels]
            ys = [p[0] for p in self._path_pixels]
            self.ax_roi.plot(xs, ys, "r-", linewidth=2, alpha=0.9)
        if self._path_length is not None:
            if self._nm_per_pixel is not None and self._nm_per_pixel > 0:
                self.ax_roi.set_title(f"{self._tr('輪郭長')}: {self._path_length * self._nm_per_pixel:.3f} nm")
            else:
                self.ax_roi.set_title(f"{self._tr('輪郭長')}: {self._path_length:.3f} px")
        else:
            self.ax_roi.set_title(self._roi_instruction_text())
        self.canvas.draw_idle()

    def run_trace_and_length(self, roi_image: np.ndarray, on_length_computed) -> None:
        """Compute path and length from [start, waypoints..., end]. Callback(length_px) when done."""
        if len(self._endpoints) < 2 or roi_image is None or roi_image.size == 0:
            return
        points = self._endpoints
        self._anchor_points_for_trace = [(float(p[0]), float(p[1])) for p in points]
        self._snapped_anchor_points_for_trace = None
        self._persistence_path_pixels = None
        self._anchor_point_count = len(points)
        if self._analysis_mode == ANALYSIS_MODE_BEADS:
            path, length_px = compute_polyline_between_points(roi_image, points)
            self._path_pixels = path
            self._path_length = length_px if path else None
            if path:
                self._persistence_path_pixels = path
        else:
            snapped_points = snap_points_to_filament_centers(
                roi_image,
                points,
                ridge_sigma=self._ridge_sigma,
                ridge_weight=self._ridge_weight,
                search_radius_px=max(3.0, self._ridge_sigma * 4.0),
            )
            self._anchor_points_for_trace = [(float(p[0]), float(p[1])) for p in snapped_points]
            self._snapped_anchor_points_for_trace = self._anchor_points_for_trace
        if self._analysis_mode != ANALYSIS_MODE_BEADS and len(self._anchor_points_for_trace or []) == 2:
            path, length_px = compute_contour_between_points(
                roi_image,
                self._anchor_points_for_trace[0],
                self._anchor_points_for_trace[1],
                ridge_sigma=self._ridge_sigma,
                ridge_weight=self._ridge_weight,
                ridge_floor_threshold=self._ridge_floor_threshold,
                max_bending_angle_deg=self._max_bending_angle_deg,
            )
            self._path_pixels = path
            self._path_length = length_px if path else None
            if path:
                self._persistence_path_pixels = path
        elif self._analysis_mode != ANALYSIS_MODE_BEADS:
            # Concatenate segments: start -> wp1 -> ... -> end (avoid duplicate junction pixels)
            combined: List[Tuple[int, int]] = []
            trace_points = self._anchor_points_for_trace or points
            for i in range(len(points) - 1):
                seg, _ = compute_contour_between_points(
                    roi_image,
                    trace_points[i],
                    trace_points[i + 1],
                    ridge_sigma=self._ridge_sigma,
                    ridge_weight=self._ridge_weight,
                    ridge_floor_threshold=self._ridge_floor_threshold,
                    max_bending_angle_deg=self._max_bending_angle_deg,
                )
                if seg is None or len(seg) < 2:
                    self._path_pixels = None
                    self._path_length = None
                    self._persistence_path_pixels = None
                    self._roi_image = roi_image
                    self._draw_roi_with_endpoints_and_path()
                    return
                if i == 0:
                    combined.extend(seg)
                else:
                    combined.extend(seg[1:])
            self._path_pixels = combined
            self._path_length = contour_length_spline(combined) if len(combined) >= 2 else 0.0
            self._persistence_path_pixels = combined if len(combined) >= 2 else None
        self._roi_image = roi_image
        self._draw_roi_with_endpoints_and_path()
        if on_length_computed and self._path_length is not None:
            on_length_computed(self._path_length)

    def update_view(
        self,
        frame: np.ndarray,
        roi_overlay: Optional[Dict[str, Any]] = None,
        roi_image: Optional[np.ndarray] = None,
        recorded_contours: Optional[List[Dict[str, Any]]] = None,
        current_frame_index: Optional[int] = None,
        current_file_id: Optional[str] = None,
    ) -> None:
        self.ax_afm.clear()
        self.ax_roi.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        self.ax_afm.imshow(frame, cmap=self._cmap_name, origin="lower")
        self.ax_afm.set_title(self._tr("Full AFM image (drag to set ROI)"))

        if recorded_contours and current_frame_index is not None:
            for i, rec in enumerate(recorded_contours):
                if rec.get("frame_index") != current_frame_index:
                    continue
                # Only show contours for the current file (file_id); old records without file_id match any
                rec_fid = rec.get("file_id")
                if rec_fid is not None and current_file_id is not None and rec_fid != current_file_id:
                    continue
                path_xy = rec.get("path_full_xy") or []
                if len(path_xy) < 2:
                    continue
                xs = [p[0] for p in path_xy]
                ys = [p[1] for p in path_xy]
                self.ax_afm.plot(xs, ys, "lime", linewidth=1.5, alpha=0.9)
                self.ax_afm.text(xs[0], ys[0], f" {i + 1}", color="lime", fontsize=9, va="bottom")

        if roi_overlay:
            if roi_overlay.get("shape") == "Ellipse":
                el = Ellipse(
                    (roi_overlay["cx"], roi_overlay["cy"]),
                    width=roi_overlay["rx"] * 2.0,
                    height=roi_overlay["ry"] * 2.0,
                    linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--",
                )
                self.ax_afm.add_patch(el)
            else:
                rect = Rectangle(
                    (roi_overlay["x0"], roi_overlay["y0"]),
                    roi_overlay["w"], roi_overlay["h"],
                    linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--",
                )
                self.ax_afm.add_patch(rect)

        self._roi_image = roi_image
        if roi_image is not None and roi_image.size > 0:
            self.ax_roi.imshow(roi_image, cmap=self._cmap_name, origin="lower")
            if self._path_length is not None:
                if self._nm_per_pixel is not None and self._nm_per_pixel > 0:
                    self.ax_roi.set_title(f"ROI - {self._tr('輪郭長')}: {self._path_length * self._nm_per_pixel:.3f} nm")
                else:
                    self.ax_roi.set_title(f"ROI - {self._tr('輪郭長')}: {self._path_length:.3f} px")
            else:
                self.ax_roi.set_title(self._roi_instruction_text())
            n_pt = len(self._endpoints)
            for i, (px, py) in enumerate(self._endpoints):
                color = "red" if i == 0 else ("cyan" if i == n_pt - 1 else "yellow")
                self.ax_roi.plot(
                    px, py, self._marker_shape, color=color,
                    markersize=self._marker_size, markeredgewidth=2,
                )
            snapped = self._snapped_anchor_points_for_trace or []
            if snapped and len(snapped) == n_pt:
                for sx, sy in snapped:
                    self.ax_roi.plot(
                        sx, sy, "+", color="white",
                        markersize=max(5.0, self._marker_size + 3.0),
                        markeredgewidth=1.4,
                    )
            if self._path_pixels and len(self._path_pixels) >= 2:
                xs = [p[1] for p in self._path_pixels]
                ys = [p[0] for p in self._path_pixels]
                self.ax_roi.plot(xs, ys, "r-", linewidth=2, alpha=0.9)
        else:
            self.ax_roi.set_title(self._tr("ROI (select region above)"))
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Fluctuation analysis result window
# ---------------------------------------------------------------------------

class FluctuationResultsWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        results_by_group: Dict[int, dict],
        parent=None,
        default_csv_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.results_by_group = results_by_group
        self.default_csv_dir = default_csv_dir
        self._ui_language = getattr(parent, "_ui_language", UI_LANG_JA)
        self.setWindowTitle("Filament Fluctuation Analysis")
        self.resize(1100, max(650, 320 * max(1, len(results_by_group))))

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        button_row = QtWidgets.QHBoxLayout()
        self.save_csv_btn = QtWidgets.QPushButton("CSV保存")
        self.save_csv_btn.setToolTip("曲率行列、分散/Lp_local、自己相関をグループ別CSVで保存します")
        self.save_csv_btn.clicked.connect(self._save_csv)
        button_row.addWidget(self.save_csv_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlainText(self._build_summary_text())
        self.summary_text.setMaximumHeight(min(220, 70 + 80 * max(1, len(results_by_group))))
        layout.addWidget(self.summary_text)

        self.figure = Figure(figsize=(12, max(3.5, 3.2 * max(1, len(results_by_group)))))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(max(360, 300 * max(1, len(results_by_group))))
        layout.addWidget(self.canvas, 1)
        self._plot_results()
        self.set_ui_language(self._ui_language)

    def _tr(self, text: str) -> str:
        return _ui_text(text, self._ui_language)

    def set_ui_language(self, lang: str) -> None:
        self._ui_language = UI_LANG_EN if lang == UI_LANG_EN else UI_LANG_JA
        self.setWindowTitle(self._tr("Filament Fluctuation Analysis"))
        self.save_csv_btn.setText(self._tr("CSV保存"))
        self.save_csv_btn.setToolTip(self._tr("曲率行列、分散/Lp_local、自己相関をグループ別CSVで保存します"))
        self.summary_text.setPlainText(self._build_summary_text())
        self._plot_results()

    @staticmethod
    def _format_num(value: Optional[float], suffix: str = "", precision: int = 3) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return f"{float(value):.{precision}f}{suffix}"

    @staticmethod
    def _csv_value(value: Any) -> Any:
        try:
            v = float(value)
        except Exception:
            return value
        if not np.isfinite(v):
            return ""
        return v

    def _build_summary_text(self) -> str:
        lines: List[str] = []
        for group_id, result in sorted(self.results_by_group.items()):
            kappa_var = np.asarray(result.get("kappa_var", []), dtype=np.float64)
            var_mean = _finite_mean(kappa_var)
            var_sd = _finite_std(kappa_var)
            lp_local = np.asarray(result.get("lp_local_nm", []), dtype=np.float64)
            lp_vals = lp_local[np.isfinite(lp_local) & (lp_local > 0)]
            tau_s = result.get("tau_s")
            lp_nm = result.get("lp_nm")
            l_min = result.get("L_min_nm")
            t_count = len(result.get("times_s", []))
            lines.append(f"Group {group_id}")
            if self._ui_language == UI_LANG_EN:
                lines.append(f"  Frames T: {t_count}")
                lines.append(f"  Used contour length L_min: {self._format_num(l_min, ' nm')}")
                lines.append(
                    "  <δκ²> s mean +/- SD: "
                    f"{self._format_num(var_mean, ' 1/nm²', 6)} +/- {self._format_num(var_sd, ' 1/nm²', 6)}"
                )
            else:
                lines.append(f"  フレーム数 T: {t_count}")
                lines.append(f"  使用輪郭長 L_min: {self._format_num(l_min, ' nm')}")
                lines.append(
                    "  <δκ²> s平均 ± SD: "
                    f"{self._format_num(var_mean, ' 1/nm²', 6)} ± {self._format_num(var_sd, ' 1/nm²', 6)}"
                )
            if lp_vals.size > 0:
                lp_med = float(np.nanmedian(lp_vals))
                lp_q25 = float(np.nanpercentile(lp_vals, 25))
                lp_q75 = float(np.nanpercentile(lp_vals, 75))
                lines.append(
                    "  Lp_local(s) median [IQR]: "
                    f"{self._format_num(lp_med, ' nm')} [{self._format_num(lp_q25, ' nm')}, {self._format_num(lp_q75, ' nm')}]"
                )
            else:
                lines.append("  Lp_local(s): not available" if self._ui_language == UI_LANG_EN else "  Lp_local(s): 計算不可")
            if tau_s is not None and np.isfinite(tau_s):
                label = "Relaxation time tau" if self._ui_language == UI_LANG_EN else "緩和時定数 τ"
                lines.append(f"  {label}: {float(tau_s):.3f} s")
            else:
                lines.append("  Relaxation time tau: fit unavailable" if self._ui_language == UI_LANG_EN else "  緩和時定数 τ: フィット不可")
            lp_label = "Persistence length Lp" if self._ui_language == UI_LANG_EN else "パーシステンス長 Lp"
            lines.append(f"  {lp_label}: {self._format_num(lp_nm, ' nm')}")
            warning = result.get("warning")
            if warning:
                lines.append(f"  {'Warning' if self._ui_language == UI_LANG_EN else '警告'}: {warning}")
            lines.append("")
        return "\n".join(lines).strip()

    def _plot_results(self) -> None:
        self.figure.clear()
        groups = sorted(self.results_by_group.items())
        n_groups = max(1, len(groups))
        gs = self.figure.add_gridspec(n_groups, 3, hspace=0.55, wspace=0.35)
        for row, (group_id, result) in enumerate(groups):
            s_nm = np.asarray(result.get("s_nm", []), dtype=np.float64)
            times_s = np.asarray(result.get("times_s", []), dtype=np.float64)
            kappa_all = np.asarray(result.get("kappa_all", []), dtype=np.float64)

            ax_heat = self.figure.add_subplot(gs[row, 0])
            if s_nm.size > 1 and times_s.size > 0 and kappa_all.size > 0:
                t0 = float(times_s[0])
                t1 = float(times_s[-1])
                if t0 == t1:
                    dt = 0.5
                    t0 -= dt
                    t1 += dt
                finite_kappa = kappa_all[np.isfinite(kappa_all)]
                vmax = float(np.nanpercentile(np.abs(finite_kappa), 95)) if finite_kappa.size > 0 else 1.0
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                im = ax_heat.imshow(
                    kappa_all,
                    aspect="auto",
                    origin="lower",
                    extent=[float(s_nm[0]), float(s_nm[-1]), t0, t1],
                    cmap="coolwarm",
                    vmin=-vmax,
                    vmax=vmax,
                )
                self.figure.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="κ [1/nm]")
            ax_heat.set_title(f"Group {group_id}: κ(s,t)")
            ax_heat.set_xlabel("s [nm]")
            ax_heat.set_ylabel("t [s]")

            ax_ac = self.figure.add_subplot(gs[row, 1])
            autocorr = result.get("autocorr")
            dt_s = result.get("autocorr_dt_s")
            if autocorr is not None and dt_s is not None:
                autocorr_arr = np.asarray(autocorr, dtype=np.float64)
                dt_arr = np.asarray(dt_s, dtype=np.float64)
                mask = np.isfinite(dt_arr) & np.isfinite(autocorr_arr)
                if np.any(mask):
                    ax_ac.scatter(dt_arr[mask], autocorr_arr[mask], color="tab:orange", s=22)
                fit = result.get("autocorr_fit")
                if fit is not None:
                    fit_arr = np.asarray(fit, dtype=np.float64)
                    fit_mask = np.isfinite(dt_arr) & np.isfinite(fit_arr)
                    if np.any(fit_mask):
                        ax_ac.plot(dt_arr[fit_mask], fit_arr[fit_mask], color="tab:red", linewidth=1.5)
            tau_s = result.get("tau_s")
            if tau_s is not None and np.isfinite(tau_s):
                ax_ac.set_title(f"C(Δt), τ = {float(tau_s):.2f} s")
            else:
                ax_ac.set_title("C(Δt), fit unavailable" if self._ui_language == UI_LANG_EN else "C(Δt), フィット不可")
            ax_ac.set_xlabel("Δt [s]")
            ax_ac.set_ylabel("C(Δt)")

            ax_var = self.figure.add_subplot(gs[row, 2])
            kappa_var = np.asarray(result.get("kappa_var", []), dtype=np.float64)
            kappa_var_se = np.asarray(result.get("kappa_var_se", []), dtype=np.float64)
            lp_local = np.asarray(result.get("lp_local_nm", []), dtype=np.float64)
            if s_nm.size == kappa_var.size:
                ax_var.plot(s_nm, kappa_var, color="tab:blue", linewidth=1.8, label="<δκ²(s)>")
                if kappa_var_se.size == kappa_var.size:
                    lower = kappa_var - kappa_var_se
                    upper = kappa_var + kappa_var_se
                    mask = np.isfinite(s_nm) & np.isfinite(lower) & np.isfinite(upper)
                    if np.any(mask):
                        ax_var.fill_between(s_nm[mask], lower[mask], upper[mask], color="tab:blue", alpha=0.22)
            ax_var.set_title("<δκ²(s)> / Lp_local(s)")
            ax_var.set_xlabel("s [nm]")
            ax_var.set_ylabel("variance [1/nm²]", color="tab:blue")
            ax_var.tick_params(axis="y", labelcolor="tab:blue")
            ax_lp = ax_var.twinx()
            if s_nm.size == lp_local.size:
                lp_mask = np.isfinite(s_nm) & np.isfinite(lp_local) & (lp_local > 0)
                if np.any(lp_mask):
                    ax_lp.plot(s_nm[lp_mask], lp_local[lp_mask], color="tab:orange", linewidth=1.6, label="Lp_local")
            ax_lp.set_ylabel("Lp_local [nm]", color="tab:orange")
            ax_lp.tick_params(axis="y", labelcolor="tab:orange")
        self.canvas.draw_idle()

    def _save_csv(self) -> None:
        default_dir = self.default_csv_dir or os.getcwd()
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self._tr("CSV保存"),
                f"{self._tr('デフォルト保存フォルダを作成できませんでした')}:\n{default_dir}\n\n{exc}",
            )
            default_dir = os.getcwd()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Save Fluctuation CSV", default_dir)
        if not out_dir:
            return
        try:
            for group_id, result in sorted(self.results_by_group.items()):
                safe_id = str(group_id).replace(os.sep, "_").replace("/", "_")
                base = os.path.join(out_dir, f"fluctuation_group_{safe_id}")
                self._write_kappa_all_csv(base + "_kappa_all.csv", group_id, result)
                self._write_kappa_var_csv(base + "_kappa_var.csv", group_id, result)
                self._write_autocorr_csv(base + "_autocorr.csv", group_id, result)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, self._tr("CSV保存"), f"{self._tr('CSV保存に失敗しました')}:\n{exc}")
            return
        QtWidgets.QMessageBox.information(self, self._tr("CSV保存"), f"{self._tr('CSVを保存しました')}:\n{out_dir}")

    def _write_kappa_all_csv(self, path: str, group_id: int, result: dict) -> None:
        s_nm = np.asarray(result.get("s_nm", []), dtype=np.float64)
        kappa_all = np.asarray(result.get("kappa_all", []), dtype=np.float64)
        times_s = np.asarray(result.get("times_s", []), dtype=np.float64)
        frame_indices = np.asarray(result.get("frame_indices", []), dtype=int)
        header = ["group_id", "frame_index", "time_s"] + [f"kappa_s_{s:.6g}_nm" for s in s_nm]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in range(kappa_all.shape[0]):
                frame_index = frame_indices[row] if row < frame_indices.size else ""
                time_s = times_s[row] if row < times_s.size else ""
                writer.writerow(
                    [group_id, frame_index, self._csv_value(time_s)]
                    + [self._csv_value(v) for v in kappa_all[row]]
                )

    def _write_kappa_var_csv(self, path: str, group_id: int, result: dict) -> None:
        s_nm = np.asarray(result.get("s_nm", []), dtype=np.float64)
        kappa_mean = np.asarray(result.get("kappa_mean", []), dtype=np.float64)
        kappa_var = np.asarray(result.get("kappa_var", []), dtype=np.float64)
        kappa_var_se = np.asarray(result.get("kappa_var_se", []), dtype=np.float64)
        lp_local = np.asarray(result.get("lp_local_nm", []), dtype=np.float64)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "group_id",
                "s_nm",
                "kappa_mean_1_per_nm",
                "kappa_var_1_per_nm2",
                "kappa_var_se_1_per_nm2",
                "lp_local_nm",
            ])
            for idx, s_val in enumerate(s_nm):
                writer.writerow([
                    group_id,
                    self._csv_value(s_val),
                    self._csv_value(kappa_mean[idx] if idx < kappa_mean.size else np.nan),
                    self._csv_value(kappa_var[idx] if idx < kappa_var.size else np.nan),
                    self._csv_value(kappa_var_se[idx] if idx < kappa_var_se.size else np.nan),
                    self._csv_value(lp_local[idx] if idx < lp_local.size else np.nan),
                ])

    def _write_autocorr_csv(self, path: str, group_id: int, result: dict) -> None:
        dt_s = np.asarray(result.get("autocorr_dt_s", []), dtype=np.float64)
        autocorr = np.asarray(result.get("autocorr", []), dtype=np.float64)
        fit = result.get("autocorr_fit")
        fit_arr = np.asarray(fit, dtype=np.float64) if fit is not None else np.full(dt_s.shape, np.nan)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["group_id", "delta_t_s", "autocorr", "autocorr_fit"])
            for idx, dt_val in enumerate(dt_s):
                writer.writerow([
                    group_id,
                    self._csv_value(dt_val),
                    self._csv_value(autocorr[idx] if idx < autocorr.size else np.nan),
                    self._csv_value(fit_arr[idx] if idx < fit_arr.size else np.nan),
                ])


class FilamentBoxCatalogWindow(QtWidgets.QMainWindow):
    def __init__(self, owner: "ContourLengthWindow", parent=None):
        super().__init__(parent)
        self.owner = owner
        self._ui_language = getattr(owner, "_ui_language", UI_LANG_JA)
        self.setWindowTitle("Filament Linearization Catalog")
        self.resize(900, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        button_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export Boxes")
        self.export_btn.setToolTip("summary/profile CSV、各boxのstrip CSV/NPY/PNG、軸CSVを保存します。profiles CSVにはs_nmごとの全値が入ります")
        self.export_btn.clicked.connect(self.owner._on_export_filament_boxes)
        button_row.addWidget(self.export_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.summary_label = QtWidgets.QLabel("Linearized boxes: 0")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.catalog_scroll = QtWidgets.QScrollArea()
        self.catalog_scroll.setWidgetResizable(True)
        self.catalog_container = QtWidgets.QWidget()
        self.catalog_layout = QtWidgets.QVBoxLayout(self.catalog_container)
        self.catalog_layout.setContentsMargins(8, 8, 8, 8)
        self.catalog_layout.setSpacing(10)
        self.catalog_scroll.setWidget(self.catalog_container)
        layout.addWidget(self.catalog_scroll, 1)
        self.retranslate()

    def _tr(self, text: str) -> str:
        return _ui_text(text, self._ui_language)

    def retranslate(self) -> None:
        self._ui_language = getattr(self.owner, "_ui_language", self._ui_language)
        self.setWindowTitle(self._tr("Filament Linearization Catalog"))
        self.export_btn.setText(self._tr("Export Boxes"))
        self.export_btn.setToolTip(self._tr("summary/profile CSV、各boxのstrip CSV/NPY/PNG、軸CSVを保存します。profiles CSVにはs_nmごとの全値が入ります"))
        self.redraw()

    def _clear_layout(self) -> None:
        while self.catalog_layout.count():
            item = self.catalog_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _view_width_px(self) -> int:
        viewport = self.catalog_scroll.viewport()
        if viewport is not None:
            width = int(viewport.width())
            if width > 20:
                return max(320, width - 36)
        return 760

    def redraw(self) -> None:
        self._clear_layout()
        boxes = self.owner._stored_filament_boxes
        self.summary_label.setText(self.owner._filament_box_summary_text())
        if not boxes:
            empty = QtWidgets.QLabel(self._tr("No linearized boxes"))
            empty.setAlignment(QtCore.Qt.AlignCenter)
            empty.setMinimumHeight(220)
            self.catalog_layout.addWidget(empty)
            self.catalog_layout.addStretch(1)
            return

        finite_parts = []
        for box in boxes:
            try:
                strip_arr = np.asarray(box.get("strip_image", []), dtype=np.float64)
            except Exception:
                continue
            if strip_arr.size:
                finite_parts.append(strip_arr[np.isfinite(strip_arr)])
        finite_parts = [part for part in finite_parts if part.size > 0]
        if finite_parts:
            all_values = np.concatenate(finite_parts)
            vmin = float(np.percentile(all_values, 1.0))
            vmax = float(np.percentile(all_values, 99.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        width_px = self._view_width_px()
        nm_per_display_px = self.owner._box_catalog_nm_per_display_px(width_px)
        for box in boxes:
            self.catalog_layout.addWidget(
                self.owner._make_filament_box_widget(box, vmin, vmax, width_px, nm_per_display_px)
            )
        self.catalog_layout.addStretch(1)
        self.catalog_container.adjustSize()
        self.catalog_container.updateGeometry()

    def closeEvent(self, event) -> None:
        try:
            if self.owner._filament_box_catalog_window is self:
                self.owner._filament_box_catalog_window = None
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main plugin window
# ---------------------------------------------------------------------------

class ContourLengthWindow(QtWidgets.QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._ui_language = UI_LANG_JA
        self.setWindowTitle(PLUGIN_NAME)
        self.setMinimumWidth(380)
        self.full_viz_window: Optional[ContourLengthFullImageWindow] = None
        self.last_frame: Optional[np.ndarray] = None
        self.manual_roi: Optional[Dict[str, Any]] = None
        self.roi_by_frame: Dict[int, Dict[str, Any]] = {}
        self._recorded_contours_list: List[Dict[str, Any]] = []
        self._current_fiber_group_id: Optional[int] = None
        self._next_fiber_group_id: int = 1
        self._fluctuation_results_window: Optional[FluctuationResultsWindow] = None
        self._filament_box_catalog_window: Optional[FilamentBoxCatalogWindow] = None
        self._stored_filament_boxes: List[Dict[str, Any]] = []
        self._next_filament_box_id: int = 1
        self._last_fluctuation_results_by_group: Dict[int, dict] = {}
        self._session_file_paths: List[str] = []
        self._last_ok_trace_signature: Optional[Tuple[Any, ...]] = None
        self._last_frame_index_seen: Optional[int] = None
        self._pending_frame_navigation_roi: Optional[Dict[str, Any]] = None
        self._build_ui()
        self._connect_frame_signal()
        self._show_full_image_view()
        self._last_frame_index_seen = self._get_current_frame_index()
        self._update_frame_label()
        self._update_fiber_group_label()
        self._update_recorded_table()
        self._update_filament_box_summary_label()
        self._retranslate_ui()

    def _build_ui(self) -> None:
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)

        language_row = QtWidgets.QHBoxLayout()
        self.language_label = QtWidgets.QLabel("")
        self.language_toggle_btn = QtWidgets.QPushButton("JPN / EN")
        self.language_toggle_btn.setMaximumWidth(90)
        self.language_toggle_btn.setToolTip("UI表示を日本語/英語に切り替えます")
        self.language_toggle_btn.clicked.connect(self._toggle_ui_language)
        self.help_btn = QtWidgets.QPushButton("Help")
        self.help_btn.setMaximumWidth(70)
        self.help_btn.setToolTip("操作内容と出力ファイルの説明を表示します")
        self.help_btn.clicked.connect(self._show_plugin_help)
        language_row.addWidget(self.language_label)
        language_row.addStretch()
        language_row.addWidget(self.help_btn)
        language_row.addWidget(self.language_toggle_btn)
        layout.addLayout(language_row)

        gb = QtWidgets.QGroupBox("ROI / 基本")
        grid = QtWidgets.QGridLayout(gb)
        r = 0
        self.roi_status_label = QtWidgets.QLabel("ROI未選択")
        grid.addWidget(self.roi_status_label, r, 0, 1, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("カラーマップ"), r, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "gray", "magma", "inferno", "plasma", "hot", "bone", "coolwarm"])
        self.cmap_combo.setToolTip("AFM画像の表示カラーマップ")
        self.cmap_combo.setMaximumWidth(80)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        grid.addWidget(self.cmap_combo, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("点のサイズ (pt)"), r, 0)
        self.marker_size_spin = QtWidgets.QDoubleSpinBox()
        self.marker_size_spin.setRange(2, 60)
        self.marker_size_spin.setValue(2)
        self.marker_size_spin.setSingleStep(1)
        self.marker_size_spin.setMaximumWidth(80)
        self.marker_size_spin.setToolTip("始点・終点・通過点のマーカーサイズ")
        self.marker_size_spin.valueChanged.connect(self._on_marker_style_changed)
        grid.addWidget(self.marker_size_spin, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("点の形"), r, 0)
        self.marker_shape_combo = QtWidgets.QComboBox()
        for _key, label_ja, _label_en, marker in MARKER_SHAPE_OPTIONS:
            self.marker_shape_combo.addItem(label_ja, marker)
        self.marker_shape_combo.setToolTip("始点・終点・通過点のマーカー形状")
        self.marker_shape_combo.setMaximumWidth(80)
        self.marker_shape_combo.currentIndexChanged.connect(self._on_marker_style_changed)
        grid.addWidget(self.marker_shape_combo, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("解析モード"), r, 0)
        self.analysis_mode_combo = QtWidgets.QComboBox()
        for label, mode in ANALYSIS_MODE_OPTIONS:
            self.analysis_mode_combo.addItem(label, mode)
        self.analysis_mode_combo.setToolTip("連続紐状(DNA)はリッジ追従、セグメント構造は点を直線接続")
        self.analysis_mode_combo.currentIndexChanged.connect(self._on_analysis_mode_changed)
        grid.addWidget(self.analysis_mode_combo, r, 1)
        r += 1
        self.roi_basic_group = gb

        gb_ridge = QtWidgets.QGroupBox("リッジ条件")
        self.ridge_group = gb_ridge
        grid_ridge = QtWidgets.QGridLayout(gb_ridge)
        rr = 0
        grid_ridge.addWidget(QtWidgets.QLabel("リッジ σ (px)"), rr, 0)
        self.ridge_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_sigma_spin.setRange(0.5, 10.0)
        self.ridge_sigma_spin.setValue(1.5)
        self.ridge_sigma_spin.setSingleStep(0.5)
        self.ridge_sigma_spin.setToolTip("リッジ検出に使うガウシアン幅。繊維の見かけ幅に近い値から調整します")
        self.ridge_sigma_spin.valueChanged.connect(self._on_ridge_sigma_changed)
        grid_ridge.addWidget(self.ridge_sigma_spin, rr, 1)
        rr += 1
        grid_ridge.addWidget(QtWidgets.QLabel("最大曲げ角度 (deg)"), rr, 0)
        self.max_bending_spin = QtWidgets.QDoubleSpinBox()
        self.max_bending_spin.setRange(0, 180)
        self.max_bending_spin.setValue(0)
        self.max_bending_spin.setSingleStep(5)
        self.max_bending_spin.setSpecialValueText("オフ")
        self.max_bending_spin.setToolTip("交差点で急角度を禁止。0=オフ。DNAの持続長に合わせて30–90程度を推奨")
        self.max_bending_spin.valueChanged.connect(self._on_max_bending_changed)
        grid_ridge.addWidget(self.max_bending_spin, rr, 1)
        rr += 1
        grid_ridge.addWidget(QtWidgets.QLabel("リッジ重み"), rr, 0)
        self.ridge_weight_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_weight_spin.setRange(0.5, 1.0)
        self.ridge_weight_spin.setValue(0.9)
        self.ridge_weight_spin.setSingleStep(0.05)
        self.ridge_weight_spin.setToolTip("コストでリッジを重視。高めにするとU字の内側を通りにくい")
        self.ridge_weight_spin.valueChanged.connect(self._on_ridge_weight_changed)
        grid_ridge.addWidget(self.ridge_weight_spin, rr, 1)
        rr += 1
        grid_ridge.addWidget(QtWidgets.QLabel("リッジ閾値（通過禁止）"), rr, 0)
        self.ridge_floor_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_floor_spin.setRange(0, 0.5)
        self.ridge_floor_spin.setValue(0)
        self.ridge_floor_spin.setSingleStep(0.05)
        self.ridge_floor_spin.setSpecialValueText("オフ")
        self.ridge_floor_spin.setToolTip("リッジがこの値未満のピクセルを通れなくする。0=オフ。端点付近は除外")
        self.ridge_floor_spin.valueChanged.connect(self._on_ridge_floor_changed)
        grid_ridge.addWidget(self.ridge_floor_spin, rr, 1)
        rr += 1
        layout.addWidget(gb_ridge)

        gb_pre = QtWidgets.QGroupBox("前処理（測長用ROI）")
        grid_pre = QtWidgets.QGridLayout(gb_pre)
        rp = 0
        self.noise_check = QtWidgets.QCheckBox("ノイズ軽減（メディアン）")
        self.noise_check.setToolTip("測長前にメディアンフィルタでノイズを軽減")
        self.noise_check.stateChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.noise_check, rp, 0, 1, 2)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("カーネル k"), rp, 0)
        self.median_spin = QtWidgets.QSpinBox()
        self.median_spin.setRange(0, 15)
        self.median_spin.setValue(3)
        self.median_spin.setSpecialValueText("オフ")
        self.median_spin.setToolTip("0=オフ。奇数推奨（3,5,7）")
        self.median_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.median_spin, rp, 1)
        rp += 1
        self.deconv_check = QtWidgets.QCheckBox("デコンボリューション (Richardson-Lucy)")
        self.deconv_check.setToolTip("探針ボケを軽減し、重なった鎖の境界をシャープに")
        self.deconv_check.stateChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.deconv_check, rp, 0, 1, 2)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("デコンボ PSF σ (px)"), rp, 0)
        self.deconv_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.deconv_sigma_spin.setRange(0.3, 5.0)
        self.deconv_sigma_spin.setValue(1.0)
        self.deconv_sigma_spin.setSingleStep(0.2)
        self.deconv_sigma_spin.setToolTip("Richardson-LucyデコンボリューションのPSF幅。大きすぎると過補正になります")
        self.deconv_sigma_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.deconv_sigma_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("デコンボ 反復回数"), rp, 0)
        self.deconv_iter_spin = QtWidgets.QSpinBox()
        self.deconv_iter_spin.setRange(1, 50)
        self.deconv_iter_spin.setValue(10)
        self.deconv_iter_spin.setToolTip("デコンボリューションの反復回数。増やすとシャープになりますがノイズも増えます")
        self.deconv_iter_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.deconv_iter_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("背景処理"), rp, 0)
        self.flatten_combo = QtWidgets.QComboBox()
        self.flatten_combo.addItem("なし", "none")
        self.flatten_combo.addItem("平面引き", "plane")
        self.flatten_combo.addItem("ガウシアン引き", "gaussian")
        self.flatten_combo.setToolTip("ROI内の傾き・ドリフトを除去")
        self.flatten_combo.currentIndexChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.flatten_combo, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("背景 σ (px)"), rp, 0)
        self.flatten_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.flatten_sigma_spin.setRange(1.0, 100.0)
        self.flatten_sigma_spin.setValue(10.0)
        self.flatten_sigma_spin.setToolTip("ガウシアン引き時のみ。鎖幅より大きく")
        self.flatten_sigma_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.flatten_sigma_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("コントラスト強調"), rp, 0)
        self.contrast_combo = QtWidgets.QComboBox()
        self.contrast_combo.addItem("なし", "none")
        self.contrast_combo.addItem("パーセンタイル", "percentile")
        self.contrast_combo.addItem("ガンマ", "gamma")
        self.contrast_combo.setToolTip("測長前のコントラスト調整")
        self.contrast_combo.currentIndexChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.contrast_combo, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("パーセンタイル Low"), rp, 0)
        self.contrast_low_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_low_spin.setRange(0, 100)
        self.contrast_low_spin.setValue(2.0)
        self.contrast_low_spin.setToolTip("パーセンタイル強調時の下限。これ以下を暗側に割り当てます")
        grid_pre.addWidget(self.contrast_low_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("パーセンタイル High"), rp, 0)
        self.contrast_high_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_high_spin.setRange(0, 100)
        self.contrast_high_spin.setValue(98.0)
        self.contrast_high_spin.setToolTip("パーセンタイル強調時の上限。これ以上を明側に割り当てます")
        grid_pre.addWidget(self.contrast_high_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("ガンマ"), rp, 0)
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.2, 3.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setToolTip("ガンマ補正値。1で無補正、1未満で暗部を強調します")
        grid_pre.addWidget(self.gamma_spin, rp, 1)
        rp += 1
        self.contrast_low_spin.valueChanged.connect(self._refresh_view)
        self.contrast_high_spin.valueChanged.connect(self._refresh_view)
        self.gamma_spin.valueChanged.connect(self._refresh_view)
        layout.addWidget(gb_pre)

        gb2 = QtWidgets.QGroupBox("輪郭長")
        v2 = QtWidgets.QVBoxLayout(gb2)
        instr_label = QtWidgets.QLabel("ROI画像上で左クリックで始点・終点。U字などは Shift+Click で通過点を追加")
        instr_label.setWordWrap(True)
        instr_label.setStyleSheet("font-size: 9px;")
        v2.addWidget(instr_label)
        record_btn_row = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.setMaximumWidth(60)
        self.ok_btn.clicked.connect(lambda: self._on_ok_record())
        self.ok_btn.setToolTip("同じファイル・同じフレーム・同じ繊維グループの既存記録がある場合は、古い記録を上書きします")
        record_btn_row.addWidget(self.ok_btn)
        self.clear_points_btn = QtWidgets.QPushButton("Clear Points")
        self.clear_points_btn.setMaximumWidth(120)
        self.clear_points_btn.setToolTip("クリック済みの始点・終点・通過点を消去します")
        self.clear_points_btn.clicked.connect(self._on_clear_trace_points)
        record_btn_row.addWidget(self.clear_points_btn)
        record_btn_row.addStretch()
        v2.addLayout(record_btn_row)
        self.ok_add_group_check = QtWidgets.QCheckBox("OKで現在グループに追加")
        self.ok_add_group_check.setChecked(True)
        self.ok_add_group_check.setToolTip("ONの場合、OKで記録すると現在の繊維グループIDも保存します")
        v2.addWidget(self.ok_add_group_check)
        self.length_label = QtWidgets.QLabel("輪郭長: — nm")
        self.length_label.setStyleSheet("font-weight: bold;")
        v2.addWidget(self.length_label)
        self.point_count_label = QtWidgets.QLabel("点数: —")
        v2.addWidget(self.point_count_label)
        self.end_to_end_label = QtWidgets.QLabel("End-to-End: —")
        v2.addWidget(self.end_to_end_label)
        self.persistence_check = QtWidgets.QCheckBox("Persistence Length も計算して記録")
        self.persistence_check.setChecked(True)
        self.persistence_check.setToolTip("ON時のみ Persistence Length を計算し、表示・記録します")
        self.persistence_check.stateChanged.connect(self._on_persistence_toggle_changed)
        v2.addWidget(self.persistence_check)
        self.persistence_label = QtWidgets.QLabel("Persistence Length: —")
        v2.addWidget(self.persistence_label)
        self.contour_length_group = gb2

        self.frame_label = QtWidgets.QLabel("Frame: —")
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.prev_btn.setToolTip("前のフレームへ移動します。ROIとクリック位置は残し、計算済み輪郭だけ消去します")
        self.prev_btn.clicked.connect(self._prev_frame)
        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.setToolTip("次のフレームへ移動します。ROIとクリック位置は残し、計算済み輪郭だけ消去します")
        self.next_btn.clicked.connect(self._next_frame)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.setAlignment(QtCore.Qt.AlignLeft)
        self.frame_nav_widget = QtWidgets.QWidget()
        frame_nav_layout = QtWidgets.QVBoxLayout(self.frame_nav_widget)
        frame_nav_layout.setContentsMargins(6, 0, 0, 0)
        frame_nav_layout.addWidget(self.frame_label)
        frame_nav_layout.addLayout(btn_row)

        gb_table = QtWidgets.QGroupBox("記録した測定値")
        table_layout = QtWidgets.QVBoxLayout(gb_table)
        self.recorded_table = QtWidgets.QTableWidget(0, 6)
        self.recorded_table.setHorizontalHeaderLabels(
            ["ID", "モード", "点数", "輪郭長", "End-to-End", "Persistence Length"]
        )
        self.recorded_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.recorded_table.setAlternatingRowColors(True)
        self.recorded_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.recorded_table.customContextMenuRequested.connect(self._on_recorded_table_context_menu)
        copy_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence.Copy, self.recorded_table, self._copy_table_to_clipboard
        )
        copy_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.recorded_table)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        scroll.setMaximumHeight(220)
        table_layout.addWidget(scroll)
        self.recorded_csv_btn = QtWidgets.QPushButton("CSV Save")
        self.recorded_csv_btn.setToolTip("記録した測定値表の全行をCSV保存します。初期フォルダは現在のAFMデータフォルダです")
        self.recorded_csv_btn.clicked.connect(self._on_save_recorded_measurements_csv)
        table_layout.addWidget(self.recorded_csv_btn)
        layout.addWidget(gb_table)

        gb_box = QtWidgets.QGroupBox("直線化 / Linearization")
        box_layout = QtWidgets.QVBoxLayout(gb_box)
        box_ctrl = QtWidgets.QHBoxLayout()
        self.linearization_btn = QtWidgets.QPushButton("Linearization")
        self.linearization_btn.setToolTip("記録済み輪郭から全フレームの直線化boxを一括作成し、カタログを開きます")
        self.linearization_btn.clicked.connect(self._on_linearize_records)
        box_ctrl.addWidget(self.linearization_btn)
        self.box_half_width_label = QtWidgets.QLabel("Box half width")
        self.box_half_width_label.setToolTip("中心線の左右に取る半幅です。box全幅はおよそこの値の2倍になります")
        box_ctrl.addWidget(self.box_half_width_label)
        self.box_half_width_spin = QtWidgets.QDoubleSpinBox()
        self.box_half_width_spin.setRange(1.0, 1000.0)
        self.box_half_width_spin.setDecimals(1)
        self.box_half_width_spin.setSingleStep(1.0)
        self.box_half_width_spin.setValue(10.0)
        self.box_half_width_spin.setSuffix(" nm")
        self.box_half_width_spin.setToolTip("中心線の左右に取る半幅です。box全幅はおよそこの値の2倍になります")
        box_ctrl.addWidget(self.box_half_width_spin)
        self.box_plane_correct_check = QtWidgets.QCheckBox("Box内1次傾き補正")
        self.box_plane_correct_check.setChecked(False)
        self.box_plane_correct_check.setToolTip("Linearization後の各box内で一次面 z = a*s + b*offset + c をfitして差し引きます。")
        box_ctrl.addWidget(self.box_plane_correct_check)
        self.open_box_catalog_btn = QtWidgets.QPushButton("Catalog")
        self.open_box_catalog_btn.setToolTip("作成済みの直線化boxカタログを開きます。再計算や保存は行いません")
        self.open_box_catalog_btn.clicked.connect(self._show_filament_box_catalog_window)
        box_ctrl.addWidget(self.open_box_catalog_btn)
        box_layout.addLayout(box_ctrl)

        self.box_summary_label = QtWidgets.QLabel("Linearized boxes: 0")
        self.box_summary_label.setWordWrap(True)
        box_layout.addWidget(self.box_summary_label)
        layout.addWidget(gb_box)

        gb_fluc = QtWidgets.QGroupBox("揺らぎ解析")
        grid_fluc = QtWidgets.QGridLayout(gb_fluc)
        rf = 0
        grid_fluc.addWidget(QtWidgets.QLabel("繊維グループ:"), rf, 0)
        self.fiber_group_label = QtWidgets.QLabel("未割り当て")
        self.fiber_group_label.setWordWrap(True)
        grid_fluc.addWidget(self.fiber_group_label, rf, 1)
        rf += 1
        self.new_fiber_group_btn = QtWidgets.QPushButton("新規グループ開始")
        self.new_fiber_group_btn.setToolTip("新しい繊維グループIDを作成し、以後のOK記録の割り当て先にします")
        self.new_fiber_group_btn.clicked.connect(self._on_new_fiber_group)
        grid_fluc.addWidget(self.new_fiber_group_btn, rf, 0, 1, 2)
        rf += 1
        self.add_to_group_btn = QtWidgets.QPushButton("現在グループに追加")
        self.add_to_group_btn.setToolTip("現在のトレースをOKと同じ処理で記録し、現在グループに割り当てます")
        self.add_to_group_btn.clicked.connect(self._on_add_current_group)
        grid_fluc.addWidget(self.add_to_group_btn, rf, 0, 1, 2)
        rf += 1
        self.delete_fiber_group_btn = QtWidgets.QPushButton("現在グループ削除")
        self.delete_fiber_group_btn.setToolTip("現在グループの割り当てを解除します。測定レコードやbox自体は削除しません")
        self.delete_fiber_group_btn.clicked.connect(self._on_delete_current_fiber_group)
        grid_fluc.addWidget(self.delete_fiber_group_btn, rf, 0, 1, 2)
        rf += 1
        self.fiber_group_info_label = QtWidgets.QLabel("グループ情報: —")
        self.fiber_group_info_label.setWordWrap(True)
        self.fiber_group_info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid_fluc.addWidget(self.fiber_group_info_label, rf, 0, 1, 2)
        rf += 1
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        grid_fluc.addWidget(sep, rf, 0, 1, 2)
        rf += 1
        grid_fluc.addWidget(QtWidgets.QLabel("フレームレート (fps)"), rf, 0)
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 100.0)
        self.fps_spin.setValue(0.5)
        self.fps_spin.setSingleStep(0.1)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setToolTip("frame_indexを秒に変換するフレームレートです。0.5 fpsなら1フレーム2秒です")
        grid_fluc.addWidget(self.fps_spin, rf, 1)
        rf += 1
        grid_fluc.addWidget(QtWidgets.QLabel("リサンプル点数"), rf, 0)
        self.n_resample_spin = QtWidgets.QSpinBox()
        self.n_resample_spin.setRange(20, 500)
        self.n_resample_spin.setValue(100)
        self.n_resample_spin.setToolTip("各輪郭を等弧長にそろえる点数です。大きいほど細かくなりますがノイズも拾いやすくなります")
        grid_fluc.addWidget(self.n_resample_spin, rf, 1)
        rf += 1
        self.run_fluctuation_btn = QtWidgets.QPushButton("揺らぎ解析を実行")
        self.run_fluctuation_btn.setToolTip("fiber_group_idが付いた記録をグループごとに曲率ゆらぎ解析します")
        self.run_fluctuation_btn.clicked.connect(self._on_run_fluctuation_analysis)
        grid_fluc.addWidget(self.run_fluctuation_btn, rf, 0, 1, 2)
        rf += 1
        self.fluctuation_summary_label = QtWidgets.QLabel("未実行")
        self.fluctuation_summary_label.setWordWrap(True)
        grid_fluc.addWidget(self.fluctuation_summary_label, rf, 0, 1, 2)
        layout.addWidget(gb_fluc)

        layout.addStretch()
        self.data_clear_btn = QtWidgets.QPushButton("Data Clear")
        self.data_clear_btn.setToolTip("記録した測定値と輪郭オーバーレイを消去します。直線化box、Sessionファイル、Export済みファイルは削除しません")
        self.data_clear_btn.clicked.connect(self._on_data_clear)
        layout.addWidget(self.data_clear_btn)
        self.save_session_btn = QtWidgets.QPushButton("Save Session")
        self.save_session_btn.setToolTip("記録済み輪郭、グループ、ROI、直線化box、解析設定をJSON保存します")
        self.save_session_btn.clicked.connect(self._save_session)
        layout.addWidget(self.save_session_btn)
        self.load_session_btn = QtWidgets.QPushButton("Load Session")
        self.load_session_btn.setToolTip("Session JSONを読み込み、途中までの解析状態を復元します")
        self.load_session_btn.clicked.connect(self._load_session)
        layout.addWidget(self.load_session_btn)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        self._retranslate_ui()

    def _tr(self, text: str) -> str:
        return _ui_text(text, self._ui_language)

    def _toggle_ui_language(self) -> None:
        self._ui_language = UI_LANG_EN if self._ui_language == UI_LANG_JA else UI_LANG_JA
        self._retranslate_ui()

    def _show_plugin_help(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Filament Analysis Help")
        dialog.resize(720, 560)
        layout = QtWidgets.QVBoxLayout(dialog)

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

        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        layout.addWidget(text, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        def set_lang(lang: str) -> None:
            use_ja = lang == UI_LANG_JA
            btn_ja.setChecked(use_ja)
            btn_en.setChecked(not use_ja)
            btn_ja.setStyleSheet(selected_style if use_ja else normal_style)
            btn_en.setStyleSheet(selected_style if not use_ja else normal_style)
            dialog.setWindowTitle("Filament Analysisヘルプ" if use_ja else "Filament Analysis Help")
            text.setPlainText(self._plugin_help_text(lang))

        btn_ja.clicked.connect(lambda: set_lang(UI_LANG_JA))
        btn_en.clicked.connect(lambda: set_lang(UI_LANG_EN))
        set_lang(UI_LANG_EN)
        dialog.exec_()

    def _plugin_help_text(self, lang: Optional[str] = None) -> str:
        active_lang = lang if lang in (UI_LANG_JA, UI_LANG_EN) else self._ui_language
        if active_lang == UI_LANG_EN:
            return (
                "Filament Analysis Help\n"
                "\n"
                "Contour recording\n"
                "- OK: records the current traced contour into the Recorded Measurements table. If clicked points remain but the contour was cleared after moving frames, OK recalculates the contour on the current frame first. If a record already exists for the same file, frame, and fiber group, OK overwrites the old record instead of adding a duplicate.\n"
                "- Clear Points: clears clicked start/end/waypoint positions and the calculated contour. Use this when filament drift makes the previous positions unsuitable.\n"
                "- Persistence Length: when checked, OK also calculates and stores the persistence length.\n"
                "- Prev / Next: moves between frames. The ROI and clicked positions are kept, while the calculated contour, length, snapped points, and persistence result are cleared for the new frame. Next asks for confirmation if a trace exists but has not been recorded with OK.\n"
                "- CSV Save under Recorded Measurements: saves all visible recorded measurement rows as CSV in the current AFM data folder by default.\n"
                "\n"
                "Tracing and preprocessing\n"
                "- Ridge sigma: Gaussian scale used for ridge detection. Start near the apparent filament width.\n"
                "- Max bending angle: prevents sharp path turns at crossings. 0 disables the limit.\n"
                "- Ridge weight / Ridge floor threshold: control how strongly the path follows ridge and intensity scores.\n"
                "- Median, deconvolution, flattening, contrast: affect only the ROI image used for tracing, not the original AFM data file.\n"
                "\n"
                "Groups and fluctuation analysis\n"
                "- Start New Group: creates a new fiber group ID for the same filament traced across frames or files.\n"
                "- Add to Current Group: records the current trace using the OK workflow and assigns it to the current group.\n"
                "- Delete Current Group: removes the group assignment from records/boxes/results. It does not delete the measurement records or linearized boxes themselves.\n"
                "- Frame rate (fps): converts frame_index to time in seconds. For 0.5 fps, one frame is 2 s.\n"
                "- Resample points: number of equal-arc-length points used for curvature calculation.\n"
                "- Run Fluctuation Analysis: calculates kappa(s,t), <delta kappa^2(s)>, Lp_local(s), autocorrelation, tau, and Lp for assigned groups.\n"
                "- Save CSV in the fluctuation result window: writes per-group kappa_all, kappa_var/Lp_local, and autocorrelation CSV files.\n"
                "\n"
                "Linearization\n"
                "- Box half width: half width sampled to the left and right of the traced centerline. The full strip width is approximately twice this value.\n"
                "- First-order box tilt correction: fits z = a*s + b*offset + c in each linearized box and subtracts it before profile calculation.\n"
                "- Linearization: loads the ASD frames referenced by the recorded contours, creates straightened boxes for all records at once, calculates height/FWHM/background/peak-offset profiles along s_nm, and opens the catalog.\n"
                "- Catalog: opens the existing linearized box catalog. It does not recalculate boxes or save files.\n"
                "- Export Boxes: saves all linearization outputs. The default folder is <current AFM data folder>/<AFM file name>/Linearization. It writes *_filament_box_summary.csv with per-box averages, *_filament_box_profiles.csv with every per-s_nm height/FWHM/background/peak-offset value, each box strip as CSV/NPY/PNG, and *_strip_axes.csv for s_nm and offset_nm axes.\n"
                "\n"
                "Session\n"
                "- Save Session: saves recorded contours, groups, ROI, linearized boxes, strip images, profiles, and analysis settings to JSON.\n"
                "- Load Session: restores the saved work state. Saved strip images can be shown even before recalculating from the original ASD file.\n"
                "- Data Clear: clears recorded measurements and contour overlays only. It does not delete session files or exported files.\n"
            )
        return (
            "Filament Analysis ヘルプ\n"
            "\n"
            "輪郭の記録\n"
            "- OK: 現在トレースした輪郭を「記録した測定値」表に追加します。Next後などでクリック位置だけが残り輪郭が未計算の場合は、現在フレームで輪郭を再計算してから記録します。同じファイル・同じフレーム・同じ繊維グループの記録が既にある場合は、重複追加せず古い記録を上書きします。\n"
            "- 位置クリア: クリック済みの始点・終点・通過点と計算済み輪郭を消去します。フレーム間ドリフトで前フレームの位置が使えない場合に押します。\n"
            "- Persistence Length: チェックON時、OK記録時にパーシステンス長も計算して保存します。\n"
            "- Prev / Next: フレームを移動します。ROIとクリック位置は残し、計算済み輪郭、輪郭長、スナップ位置、Persistence Lengthは新フレーム用に消去します。未記録のトレースがある状態でNextを押すと確認します。\n"
            "- 記録した測定値のCSV Save: 表に表示されている全行をCSV保存します。初期保存先は現在のAFMデータフォルダです。\n"
            "\n"
            "トレースと前処理\n"
            "- リッジ σ: リッジ検出に使うガウシアン幅です。繊維の見かけ幅に近い値から調整します。\n"
            "- 最大曲げ角度: 交差点などで急な折れ曲がりを禁止します。0で無効です。\n"
            "- リッジ重み / リッジ閾値: ridge と intensity の混合スコアに沿って経路を選ぶ強さを調整します。\n"
            "- メディアン、デコンボ、背景処理、コントラスト: 輪郭検出に使うROI画像だけに効きます。元のAFMデータファイルは変更しません。\n"
            "\n"
            "グループと揺らぎ解析\n"
            "- 新規グループ開始: 複数フレーム/ファイルで同じ繊維として扱うための新しいグループIDを作ります。\n"
            "- 現在グループに追加: 現在のトレースをOKと同じ処理で記録し、現在グループに割り当てます。\n"
            "- 現在グループ削除: レコード、box、結果からグループ割り当てを解除します。測定レコードや直線化box自体は削除しません。\n"
            "- フレームレート (fps): frame_indexを秒に変換します。0.5 fpsなら1フレームは2秒です。\n"
            "- リサンプル点数: 曲率計算前に各輪郭を等弧長でそろえる点数です。\n"
            "- 揺らぎ解析を実行: グループ割り当て済みの記録からκ(s,t)、<δκ²(s)>、Lp_local(s)、自己相関、τ、Lpを計算します。\n"
            "- 揺らぎ解析結果ウィンドウのCSV保存: グループごとにkappa_all、kappa_var/Lp_local、autocorrのCSVを保存します。\n"
            "\n"
            "Linearization\n"
            "- Box half width: トレース中心線の左右に取る半幅です。strip画像の全幅はおよそこの2倍です。\n"
            "- Box内1次傾き補正: 各直線化box内で z = a*s + b*offset + c をfitして差し引き、その補正後画像からプロファイルを計算します。\n"
            "- Linearization: 記録済み輪郭が参照するASDフレームを読み込み、全レコードを一括で直線化boxにします。s_nmごとの高さ、FWHM、背景値、ピークoffsetを計算し、カタログを開きます。\n"
            "- Catalog: 既に作成済みの直線化boxカタログを開きます。再計算や保存はしません。\n"
            "- Export Boxes: 直線化結果をまとめて保存します。初期保存先は <現在のAFMデータフォルダ>/<AFMファイル名>/Linearization です。*_filament_box_summary.csv にはboxごとの平均/中央値など、*_filament_box_profiles.csv にはs_nmごとの高さ/FWHM/背景/ピークoffsetの全値、各boxのstrip画像はCSV/NPY/PNG、軸情報は *_strip_axes.csv として保存します。\n"
            "\n"
            "Session\n"
            "- Save Session: 記録済み輪郭、グループ、ROI、直線化box、strip画像、profile、解析設定をJSON保存します。\n"
            "- Load Session: 保存した作業状態を復元します。保存済みstrip画像は、元ASDから再計算する前でも表示できます。\n"
            "- Data Clear: 記録した測定値と輪郭オーバーレイだけを消去します。SessionファイルやExport済みファイルは削除しません。\n"
        )

    def _translate_widget_tree(self, root: QtWidgets.QWidget) -> None:
        widgets = [root] + list(root.findChildren(QtWidgets.QWidget))
        for widget in widgets:
            if isinstance(widget, QtWidgets.QGroupBox):
                widget.setTitle(self._tr(widget.title()))
            if isinstance(widget, QtWidgets.QAbstractButton):
                widget.setText(self._tr(widget.text()))
                if widget.toolTip():
                    widget.setToolTip(self._tr(widget.toolTip()))
            elif isinstance(widget, QtWidgets.QLabel):
                widget.setText(self._tr(widget.text()))
                if widget.toolTip():
                    widget.setToolTip(self._tr(widget.toolTip()))
            elif isinstance(widget, QtWidgets.QComboBox):
                if widget.toolTip():
                    widget.setToolTip(self._tr(widget.toolTip()))
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                if widget.toolTip():
                    widget.setToolTip(self._tr(widget.toolTip()))

    def _set_combo_texts_by_data(self, combo: QtWidgets.QComboBox, labels: Dict[Any, Tuple[str, str]]) -> None:
        current = combo.currentData()
        blocker = QtCore.QSignalBlocker(combo)
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            label_pair = labels.get(data)
            if label_pair is None:
                continue
            combo.setItemText(idx, label_pair[0] if self._ui_language == UI_LANG_JA else label_pair[1])
        match = combo.findData(current)
        if match >= 0:
            combo.setCurrentIndex(match)
        del blocker

    def _analysis_mode_display_label(self, mode: str) -> str:
        if mode == ANALYSIS_MODE_BEADS:
            return self._tr("セグメント構造")
        return self._tr("連続紐状")

    def _set_recorded_table_headers(self) -> None:
        if not hasattr(self, "recorded_table"):
            return
        self.recorded_table.setHorizontalHeaderLabels([
            "ID",
            self._tr("モード"),
            self._tr("点数"),
            self._tr("輪郭長"),
            "End-to-End",
            "Persistence Length",
        ])

    def _retranslate_ui(self) -> None:
        if not hasattr(self, "language_toggle_btn"):
            return
        self.setWindowTitle(PLUGIN_NAME)
        self._translate_widget_tree(self)
        if self.full_viz_window is not None:
            self._translate_widget_tree(self.full_viz_window)
            self.full_viz_window.set_ui_language(self._ui_language)
        if self._filament_box_catalog_window is not None:
            self._filament_box_catalog_window.retranslate()
        if self._fluctuation_results_window is not None:
            self._fluctuation_results_window.set_ui_language(self._ui_language)

        self._set_combo_texts_by_data(
            self.marker_shape_combo,
            {marker: (label_ja, label_en) for _key, label_ja, label_en, marker in MARKER_SHAPE_OPTIONS},
        )
        self._set_combo_texts_by_data(
            self.analysis_mode_combo,
            {
                ANALYSIS_MODE_DNA: ("連続紐状 (DNA/リッジ追従)", "Continuous filament (DNA/ridge tracing)"),
                ANALYSIS_MODE_BEADS: ("セグメント構造（点指定）", "Segmented structure (point-specified)"),
            },
        )
        self._set_combo_texts_by_data(
            self.flatten_combo,
            {
                "none": ("なし", "None"),
                "plane": ("平面引き", "Plane subtraction"),
                "gaussian": ("ガウシアン引き", "Gaussian subtraction"),
            },
        )
        self._set_combo_texts_by_data(
            self.contrast_combo,
            {
                "none": ("なし", "None"),
                "percentile": ("パーセンタイル", "Percentile"),
                "gamma": ("ガンマ", "Gamma"),
            },
        )

        self.max_bending_spin.setSpecialValueText(self._tr("オフ"))
        self.ridge_floor_spin.setSpecialValueText(self._tr("オフ"))
        self.median_spin.setSpecialValueText(self._tr("オフ"))
        current_language = self._tr("Japanese") if self._ui_language == UI_LANG_JA else self._tr("English")
        self.language_label.setText(f"{self._tr('Language')}: {current_language}")
        self.language_toggle_btn.setText("JPN / EN")
        if hasattr(self, "clear_points_btn"):
            self.clear_points_btn.setText(self._tr("Clear Points"))
            self.clear_points_btn.setToolTip(self._tr("クリック済みの始点・終点・通過点を消去します"))
        self._set_recorded_table_headers()
        self._update_recorded_table()
        self._update_frame_label()
        self._update_fiber_group_label()
        self._update_filament_box_summary_label()
        if hasattr(self, "fluctuation_summary_label") and self.fluctuation_summary_label.text() in ("未実行", "Not run"):
            self.fluctuation_summary_label.setText(self._tr("未実行"))
        self._refresh_view()

    def _update_fiber_group_label(self) -> None:
        if not hasattr(self, "fiber_group_label"):
            return
        if self._current_fiber_group_id is None:
            self.fiber_group_label.setText(self._tr("未割り当て"))
        else:
            if self._ui_language == UI_LANG_EN:
                self.fiber_group_label.setText(f"Assigning to group {self._current_fiber_group_id}")
            else:
                self.fiber_group_label.setText(f"グループ {self._current_fiber_group_id} に割り当て中")
        self._update_fiber_group_info_label()

    def _display_file_name(self, file_id: Any) -> str:
        if file_id is None or file_id == "":
            return "unknown"
        text = str(file_id)
        base = os.path.basename(text)
        return base or text

    def _fiber_group_summary_lines(self) -> List[str]:
        group_records: Dict[int, List[dict]] = {}
        for rec in self._recorded_contours_list:
            group_id = rec.get("fiber_group_id")
            if group_id is None:
                continue
            try:
                group_id_int = int(group_id)
            except Exception:
                continue
            group_records.setdefault(group_id_int, []).append(rec)

        known_ids = set(group_records.keys())
        if self._current_fiber_group_id is not None:
            known_ids.add(int(self._current_fiber_group_id))
        for box in self._stored_filament_boxes:
            group_id = box.get("fiber_group_id")
            if group_id is None:
                continue
            try:
                known_ids.add(int(group_id))
            except Exception:
                continue

        lines: List[str] = []
        current_file = self._display_file_name(self._get_current_file_id())
        for group_id in sorted(known_ids):
            records = group_records.get(group_id, [])
            files = sorted({
                self._display_file_name(rec.get("file_id"))
                for rec in records
                if rec.get("file_id") is not None
            })
            if not files and group_id == self._current_fiber_group_id:
                files = [current_file]
            frame_numbers = sorted({
                int(rec.get("frame_index", 0)) + 1
                for rec in records
                if rec.get("frame_index") is not None
            })
            box_count = 0
            for box in self._stored_filament_boxes:
                try:
                    if box.get("fiber_group_id") is not None and int(box.get("fiber_group_id")) == group_id:
                        box_count += 1
                except Exception:
                    continue
            current_mark = " *" if group_id == self._current_fiber_group_id else ""
            file_text = ", ".join(files[:3]) if files else "—"
            if len(files) > 3:
                file_text += f" +{len(files) - 3} more" if self._ui_language == UI_LANG_EN else f" 他{len(files) - 3}"
            frame_text = ", ".join(str(v) for v in frame_numbers[:8]) if frame_numbers else "—"
            if len(frame_numbers) > 8:
                frame_text += f" +{len(frame_numbers) - 8} more" if self._ui_language == UI_LANG_EN else f" 他{len(frame_numbers) - 8}"
            lines.append(
                f"G{group_id}{current_mark}: records {len(records)}, boxes {box_count}, "
                f"files [{file_text}], frames [{frame_text}]"
            )
        return lines

    def _update_fiber_group_info_label(self) -> None:
        if not hasattr(self, "fiber_group_info_label"):
            return
        lines = self._fiber_group_summary_lines()
        if not lines:
            self.fiber_group_info_label.setText(f"{self._tr('グループ情報')}: —")
        else:
            self.fiber_group_info_label.setText(f"{self._tr('グループ情報')}:\n" + "\n".join(lines))

    def _on_new_fiber_group(self) -> None:
        self._current_fiber_group_id = self._next_fiber_group_id
        self._next_fiber_group_id += 1
        self._update_fiber_group_label()

    def _on_add_current_group(self) -> None:
        if self._current_fiber_group_id is None:
            QtWidgets.QMessageBox.information(self, self._tr("揺らぎ解析"), self._tr("先に新規グループを開始してください。"))
            return
        self._on_ok_record(assign_current_group=True)

    def _on_delete_current_fiber_group(self) -> None:
        group_id = self._current_fiber_group_id
        if group_id is None:
            QtWidgets.QMessageBox.information(self, self._tr("揺らぎ解析"), self._tr("削除する現在グループがありません。"))
            return
        group_id = int(group_id)
        record_count = 0
        for rec in self._recorded_contours_list:
            try:
                if rec.get("fiber_group_id") is not None and int(rec.get("fiber_group_id")) == group_id:
                    record_count += 1
            except Exception:
                continue
        box_count = 0
        for box in self._stored_filament_boxes:
            try:
                if box.get("fiber_group_id") is not None and int(box.get("fiber_group_id")) == group_id:
                    box_count += 1
            except Exception:
                continue
        has_result = group_id in self._last_fluctuation_results_by_group
        if record_count or box_count or has_result:
            msg = (
                (
                    f"Remove assignment for group {group_id}?\n"
                    f"records: {record_count}, boxes: {box_count}\n\n"
                    f"{self._tr('測定レコードと直線化box自体は削除しません。')}"
                )
                if self._ui_language == UI_LANG_EN else
                f"グループ {group_id} の割り当てを解除しますか？\n"
                f"record: {record_count} 件, box: {box_count} 件\n\n"
                "測定レコードと直線化box自体は削除しません。"
            )
            ret = QtWidgets.QMessageBox.question(
                self,
                self._tr("現在グループ削除"),
                msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        for rec in self._recorded_contours_list:
            try:
                if rec.get("fiber_group_id") is not None and int(rec.get("fiber_group_id")) == group_id:
                    rec["fiber_group_id"] = None
            except Exception:
                continue
        for box in self._stored_filament_boxes:
            try:
                if box.get("fiber_group_id") is not None and int(box.get("fiber_group_id")) == group_id:
                    box["fiber_group_id"] = None
            except Exception:
                continue
        self._last_fluctuation_results_by_group.pop(group_id, None)
        if not record_count and not box_count and not has_result and group_id == self._next_fiber_group_id - 1:
            self._next_fiber_group_id = max(1, self._next_fiber_group_id - 1)
        self._current_fiber_group_id = None
        self._update_fiber_group_label()
        self._update_recorded_table()
        self._redraw_filament_box_catalog()
        self._refresh_view()
        if self._last_fluctuation_results_by_group:
            self._show_fluctuation_results_window(self._last_fluctuation_results_by_group)
        elif self._fluctuation_results_window is not None:
            self._fluctuation_results_window.close()
            self._fluctuation_results_window = None
            self.fluctuation_summary_label.setText(self._tr("未実行"))

    def _current_trace_signature(self) -> Optional[Tuple[Any, ...]]:
        if self.full_viz_window is None:
            return None
        path_pixels = getattr(self.full_viz_window, "_path_pixels", None)
        endpoints = getattr(self.full_viz_window, "_endpoints", None) or []
        if path_pixels and len(path_pixels) >= 2:
            rounded_path = tuple((int(p[0]), int(p[1])) for p in path_pixels)
            return (
                self._get_current_file_id(),
                self._get_current_frame_index(),
                rounded_path,
            )
        if endpoints:
            rounded_points = tuple((round(float(p[0]), 3), round(float(p[1]), 3)) for p in endpoints)
            return (
                self._get_current_file_id(),
                self._get_current_frame_index(),
                rounded_points,
            )
        return None

    def _has_unrecorded_trace(self) -> bool:
        sig = self._current_trace_signature()
        return sig is not None and sig != self._last_ok_trace_signature

    def _confirm_next_with_unrecorded_trace(self) -> bool:
        if not self._has_unrecorded_trace():
            return True
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(self._tr("未記録の輪郭"))
        msg.setText(self._tr("OKで記録していない点または輪郭があります。Nextで次フレームへ移動しますか？"))
        msg.setInformativeText(self._tr("記録する場合はキャンセルしてOKを押してください。"))
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        msg.setButtonText(QtWidgets.QMessageBox.Yes, self._tr("移動"))
        msg.setButtonText(QtWidgets.QMessageBox.Cancel, self._tr("キャンセル"))
        return msg.exec_() == QtWidgets.QMessageBox.Yes

    def _on_run_fluctuation_analysis(self) -> None:
        nm_per_px = self._get_nm_per_pixel()
        if nm_per_px is None or nm_per_px <= 0:
            QtWidgets.QMessageBox.warning(self, self._tr("揺らぎ解析"), self._tr("nm/pxスケールを取得できません。"))
            return

        groups: Dict[int, List[dict]] = {}
        for rec in self._recorded_contours_list:
            group_id = rec.get("fiber_group_id")
            if group_id is None:
                continue
            try:
                group_id_int = int(group_id)
            except Exception:
                continue
            groups.setdefault(group_id_int, []).append(rec)
        if not groups:
            QtWidgets.QMessageBox.warning(self, self._tr("揺らぎ解析"), self._tr("割り当て済みの繊維グループがありません。"))
            return

        fps = float(self.fps_spin.value())
        n_resample = int(self.n_resample_spin.value())
        results_by_group: Dict[int, dict] = {}
        skipped: List[str] = []
        warnings: List[str] = []
        for group_id, records in sorted(groups.items()):
            if len(records) < 2:
                skipped.append(
                    f"Group {group_id}: fewer than 2 valid contour frames."
                    if self._ui_language == UI_LANG_EN else
                    f"Group {group_id}: 有効な輪郭が2フレーム未満です。"
                )
                continue
            try:
                result = fluctuation_analysis(
                    records,
                    nm_per_px=nm_per_px,
                    fps=fps,
                    n_resample=n_resample,
                    trim_frac=0.1,
                )
            except Exception as exc:
                skipped.append(f"Group {group_id}: {exc}")
                continue
            results_by_group[group_id] = result
            if result.get("warning"):
                warnings.append(f"Group {group_id}: {result['warning']}")

        if not results_by_group:
            msg = self._tr("解析可能なグループがありません。")
            if skipped:
                msg += "\n\n" + "\n".join(skipped)
            self.fluctuation_summary_label.setText(self._tr("解析不可"))
            QtWidgets.QMessageBox.warning(self, self._tr("揺らぎ解析"), msg)
            return

        self._show_fluctuation_results_window(results_by_group)
        summary_lines = [
            f"{len(results_by_group)} groups analyzed"
            if self._ui_language == UI_LANG_EN else
            f"{len(results_by_group)} グループ解析完了"
        ]
        for group_id, result in sorted(results_by_group.items()):
            tau_s = result.get("tau_s")
            lp_nm = result.get("lp_nm")
            tau_text = f"{tau_s:.3f} s" if tau_s is not None and np.isfinite(tau_s) else self._tr("フィット不可")
            lp_text = f"{lp_nm:.3f} nm" if lp_nm is not None and np.isfinite(lp_nm) else "—"
            summary_lines.append(f"Group {group_id}: τ={tau_text}, Lp={lp_text}")
        self.fluctuation_summary_label.setText("\n".join(summary_lines))

        notice_lines = warnings + skipped
        if notice_lines:
            QtWidgets.QMessageBox.warning(self, self._tr("揺らぎ解析"), "\n".join(notice_lines))

    def _show_fluctuation_results_window(self, results_by_group: Dict[int, dict]) -> None:
        self._last_fluctuation_results_by_group = dict(results_by_group)
        if self._fluctuation_results_window is not None:
            try:
                self._fluctuation_results_window.close()
            except Exception:
                pass
        self._fluctuation_results_window = FluctuationResultsWindow(
            results_by_group,
            self,
            default_csv_dir=self._default_fluctuation_csv_dir(),
        )
        self._fluctuation_results_window.show()

    def _on_linearize_records(self) -> None:
        records = [
            rec for rec in self._recorded_contours_list
            if len(rec.get("path_full_xy") or []) >= 2
        ]
        if not records:
            QtWidgets.QMessageBox.information(
                self,
                self._tr("Linearization"),
                self._tr("直線化できる記録済み輪郭がありません。"),
            )
            return

        if self._stored_filament_boxes:
            ret = QtWidgets.QMessageBox.question(
                self,
                self._tr("Linearization"),
                self._tr("既存の直線化ボックスを置き換えて、記録済み輪郭から再作成しますか？"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        half_width_nm = float(self.box_half_width_spin.value())
        plane_correct = bool(self.box_plane_correct_check.isChecked()) if hasattr(self, "box_plane_correct_check") else False
        saved_file_num = int(getattr(gv, "currentFileNum", -1))
        saved_index = int(getattr(gv, "index", 0))
        saved_file_list = list(getattr(gv, "files", []) or [])

        progress = QtWidgets.QProgressDialog(
            f"{self._tr('Linearization')}...",
            self._tr("キャンセル"),
            0,
            len(records),
            self,
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(300)

        new_boxes: List[Dict[str, Any]] = []
        skipped: List[str] = []
        frame_cache: Dict[Tuple[str, int], Tuple[np.ndarray, float, float]] = {}
        try:
            for idx, rec in enumerate(records):
                progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
                if progress.wasCanceled():
                    skipped.append(self._tr("ユーザー操作により中断しました。"))
                    break
                try:
                    box = self._linearize_record_to_box(
                        rec,
                        half_width_nm,
                        plane_correct,
                        idx + 1,
                        frame_cache,
                    )
                except Exception as exc:
                    skipped.append(f"record {idx + 1}: {exc}")
                    continue
                new_boxes.append(box)
            progress.setValue(len(records))
        finally:
            self._restore_loaded_frame_after_linearization(saved_file_num, saved_index, saved_file_list)

        if not new_boxes:
            msg = self._tr("直線化ボックスを作成できませんでした。")
            if skipped:
                msg += "\n\n" + "\n".join(skipped[:12])
            QtWidgets.QMessageBox.warning(self, self._tr("Linearization"), msg)
            return

        self._stored_filament_boxes = new_boxes
        self._next_filament_box_id = len(new_boxes) + 1
        self._redraw_filament_box_catalog()
        self._update_fiber_group_info_label()
        self._show_filament_box_catalog_window()

        msg = (
            f"{len(new_boxes)} linearized boxes were created."
            if self._ui_language == UI_LANG_EN else
            f"{len(new_boxes)} 個の直線化ボックスを作成しました。"
        )
        if skipped:
            msg += f"\n\n{self._tr('スキップ')}:\n" + "\n".join(skipped[:12])
            if len(skipped) > 12:
                msg += f"\n... +{len(skipped) - 12} more" if self._ui_language == UI_LANG_EN else f"\n... 他{len(skipped) - 12}件"
            QtWidgets.QMessageBox.warning(self, self._tr("Linearization"), msg)
        else:
            QtWidgets.QMessageBox.information(self, self._tr("Linearization"), msg)

    def _linearize_record_to_box(
        self,
        rec: Dict[str, Any],
        half_width_nm: float,
        plane_correct: bool,
        box_id: int,
        frame_cache: Dict[Tuple[str, int], Tuple[np.ndarray, float, float]],
    ) -> Dict[str, Any]:
        path_full_xy = rec.get("path_full_xy") or []
        if len(path_full_xy) < 2:
            raise ValueError("有効な輪郭座標がありません。")
        file_path = self._resolve_record_file_path(rec)
        if not file_path:
            raise ValueError("元ASDファイルを特定できません。")
        try:
            frame_index = int(rec.get("frame_index", 0))
        except Exception:
            frame_index = 0
        if frame_index < 0:
            frame_index = 0

        cache_key = (file_path, frame_index)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = self._load_frame_for_linearization(file_path, frame_index)
        frame, nm_x, nm_y = frame_cache[cache_key]
        points_yx = np.asarray([(float(y), float(x)) for x, y in path_full_xy], dtype=np.float64)
        strip, strip_s_nm, strip_offsets_nm = straighten_filament_box(
            frame,
            points_yx,
            nm_x,
            nm_y,
            half_width_nm,
        )
        if strip.size == 0 or strip_s_nm.size == 0:
            raise ValueError("直線化ボックスを作成できません。")
        if plane_correct:
            strip = subtract_linear_plane_from_strip(strip, strip_s_nm, strip_offsets_nm)
        profiles = compute_filament_box_profiles(strip, strip_offsets_nm)
        length_nm = float(strip_s_nm[-1]) if strip_s_nm.size else float("nan")
        return {
            "box_id": box_id,
            "source_path": file_path,
            "image_id": os.path.splitext(os.path.basename(file_path))[0] or str(rec.get("file_id") or "unknown"),
            "frame_index": frame_index,
            "fiber_group_id": rec.get("fiber_group_id"),
            "length_nm": length_nm,
            "box_half_width_nm": half_width_nm,
            "plane_corrected": bool(plane_correct),
            "path_full_xy": path_full_xy,
            "points_yx": points_yx,
            "strip_image": strip,
            "strip_s_nm": strip_s_nm,
            "strip_offsets_nm": strip_offsets_nm,
            "height_profile_nm": profiles["height_nm"],
            "fwhm_profile_nm": profiles["fwhm_nm"],
            "background_profile_nm": profiles["background_nm"],
            "peak_offset_nm": profiles["peak_offset_nm"],
        }

    def _resolve_record_file_path(self, rec: Dict[str, Any]) -> Optional[str]:
        candidates: List[str] = []
        file_id = rec.get("file_id")
        if file_id:
            candidates.append(str(file_id))
        candidates.extend(str(p) for p in getattr(self, "_session_file_paths", []) if p)
        candidates.extend(str(p) for p in getattr(gv, "files", []) or [] if p)

        normalized_seen = set()
        unique_candidates: List[str] = []
        for candidate in candidates:
            normalized = os.path.normpath(os.path.abspath(candidate))
            if normalized in normalized_seen:
                continue
            normalized_seen.add(normalized)
            unique_candidates.append(candidate)

        if file_id:
            file_base = os.path.basename(str(file_id))
            for candidate in unique_candidates:
                if os.path.exists(candidate) and os.path.basename(candidate) == file_base:
                    return os.path.normpath(os.path.abspath(candidate))
        for candidate in unique_candidates:
            if os.path.exists(candidate):
                return os.path.normpath(os.path.abspath(candidate))
        return None

    def _load_frame_for_linearization(self, file_path: str, frame_index: int) -> Tuple[np.ndarray, float, float]:
        if not os.path.exists(file_path):
            raise ValueError(f"ファイルが見つかりません: {file_path}")
        gv.index = int(frame_index)
        LoadFrame(file_path)
        InitializeAryDataFallback()
        frame_count = int(getattr(gv, "FrameNum", 0) or 0)
        if frame_count > 0 and frame_index >= frame_count:
            raise ValueError(f"frame_index {frame_index + 1} が範囲外です。")
        nm_xy = self._get_nm_per_pixel_xy()
        if nm_xy is None:
            raise ValueError("nm/pxスケールを取得できません。")
        if not hasattr(gv, "aryData") or gv.aryData is None:
            raise ValueError("画像データを読み込めません。")
        frame = np.asarray(gv.aryData, dtype=np.float64)
        if frame.ndim != 2:
            raise ValueError("2D画像データを取得できません。")
        return frame.copy(), float(nm_xy[0]), float(nm_xy[1])

    def _restore_loaded_frame_after_linearization(
        self,
        saved_file_num: int,
        saved_index: int,
        saved_file_list: List[str],
    ) -> None:
        try:
            if saved_file_list:
                gv.files = saved_file_list
            if 0 <= saved_file_num < len(getattr(gv, "files", []) or []):
                gv.currentFileNum = saved_file_num
                gv.index = saved_index
                if self.main_window is not None and hasattr(self.main_window, "FileList"):
                    try:
                        blocker = QtCore.QSignalBlocker(self.main_window.FileList)
                        self.main_window.FileList.setCurrentRow(saved_file_num)
                        del blocker
                    except Exception:
                        pass
                LoadFrame(gv.files[saved_file_num])
                InitializeAryDataFallback()
                if self.main_window is not None and hasattr(self.main_window, "frameSlider"):
                    try:
                        self.main_window.frameSlider.setValue(saved_index)
                    except Exception:
                        pass
                self._on_frame_changed(saved_index)
            self._update_frame_label()
        except Exception:
            pass

    def _on_export_filament_boxes(self) -> None:
        if not self._stored_filament_boxes:
            QtWidgets.QMessageBox.information(self, self._tr("Export Boxes"), self._tr("直線化ボックスがありません。"))
            return
        default_dir = self._default_linearization_export_dir()
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self._tr("Export Boxes"),
                f"{self._tr('デフォルト保存フォルダを作成できませんでした')}:\n{default_dir}\n\n{exc}",
            )
            default_dir = self._current_afm_data_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, self._tr("Export Boxes"), default_dir)
        if not out_dir:
            return
        try:
            self._export_filament_boxes(out_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, self._tr("Export Boxes"), f"{self._tr('出力に失敗しました')}:\n{exc}")
            return
        QtWidgets.QMessageBox.information(self, self._tr("Export Boxes"), f"{self._tr('直線化ボックスを出力しました')}:\n{out_dir}")

    def _current_path_full_xy(self) -> List[Tuple[float, float]]:
        if self.full_viz_window is None or self.manual_roi is None:
            return []
        path_pixels = self.full_viz_window._path_pixels or []
        if len(path_pixels) < 2:
            return []
        x0 = float(self.manual_roi.get("x0", 0))
        y0 = float(self.manual_roi.get("y0", 0))
        return [(x0 + float(p[1]), y0 + float(p[0])) for p in path_pixels]

    def _filament_box_summary_text(self) -> str:
        n = len(self._stored_filament_boxes)
        if n == 0:
            return "Linearized boxes: 0" if self._ui_language == UI_LANG_EN else "直線化box: 0"
        heights: List[float] = []
        widths: List[float] = []
        lengths: List[float] = []
        for box in self._stored_filament_boxes:
            heights.extend([float(v) for v in np.asarray(box.get("height_profile_nm", []), dtype=np.float64) if np.isfinite(v)])
            widths.extend([float(v) for v in np.asarray(box.get("fwhm_profile_nm", []), dtype=np.float64) if np.isfinite(v)])
            length_nm = box.get("length_nm")
            try:
                length_val = float(length_nm)
                if np.isfinite(length_val):
                    lengths.append(length_val)
            except Exception:
                pass
        mean_h = float(np.mean(heights)) if heights else float("nan")
        mean_w = float(np.mean(widths)) if widths else float("nan")
        total_l = float(np.sum(lengths)) if lengths else float("nan")
        if self._ui_language == UI_LANG_EN:
            return (
                f"Linearized boxes: {n}, total length {total_l:.1f} nm, "
                f"mean height {mean_h:.3g} nm, mean FWHM {mean_w:.3g} nm"
            )
        return (
            f"直線化box: {n}, 総長 {total_l:.1f} nm, "
            f"平均高さ {mean_h:.3g} nm, 平均FWHM {mean_w:.3g} nm"
        )

    def _update_filament_box_summary_label(self) -> None:
        if hasattr(self, "box_summary_label"):
            self.box_summary_label.setText(self._filament_box_summary_text())

    def _show_filament_box_catalog_window(self) -> None:
        if self._filament_box_catalog_window is None:
            self._filament_box_catalog_window = FilamentBoxCatalogWindow(self, self)
        else:
            self._filament_box_catalog_window.redraw()
        self._filament_box_catalog_window.show()
        self._filament_box_catalog_window.raise_()
        self._filament_box_catalog_window.activateWindow()

    def _redraw_filament_box_catalog(self) -> None:
        self._update_filament_box_summary_label()
        if self._filament_box_catalog_window is not None:
            self._filament_box_catalog_window.redraw()

    def _clear_box_catalog_layout(self) -> None:
        return

    def _box_catalog_view_width_px(self) -> int:
        return 760

    def _box_catalog_nm_per_display_px(self, max_width_px: int) -> float:
        max_s = 1.0
        for box in self._stored_filament_boxes:
            s_nm = np.asarray(box.get("strip_s_nm", []), dtype=np.float64)
            if s_nm.size >= 2:
                extent = abs(float(s_nm[-1]) - float(s_nm[0]))
                if np.isfinite(extent):
                    max_s = max(max_s, extent)
        return max_s / max(1, int(max_width_px))

    def _make_filament_box_widget(
        self,
        box: Dict[str, Any],
        vmin: float,
        vmax: float,
        text_width: int,
        nm_per_px: float,
    ) -> QtWidgets.QWidget:
        widget = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        box_id = int(box.get("box_id", 0))
        frame_text = f"F{int(box.get('frame_index', 0)) + 1}"
        image_id = str(box.get("image_id") or "unknown")
        length_nm = float(box.get("length_nm", float("nan")))
        mean_height = _finite_mean(np.asarray(box.get("height_profile_nm", []), dtype=np.float64))
        mean_fwhm = _finite_mean(np.asarray(box.get("fwhm_profile_nm", []), dtype=np.float64))
        title = (
            f"#{box_id} {image_id} {frame_text}, "
            + (
                f"length {length_nm:.1f} nm, height {mean_height:.3g} nm, FWHM {mean_fwhm:.3g} nm"
                if self._ui_language == UI_LANG_EN else
                f"長さ {length_nm:.1f} nm, 高さ {mean_height:.3g} nm, FWHM {mean_fwhm:.3g} nm"
            )
        )
        title_label = QtWidgets.QLabel(title)
        title_label.setWordWrap(True)
        title_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(title_label)

        strip = np.asarray(box.get("strip_image", []), dtype=np.float64)
        image_width = text_width
        image_height = 58
        image_label = QtWidgets.QLabel()
        image_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        if strip.size:
            rgb = _strip_to_rgb(strip, vmin, vmax)
            pixmap = _rgb_to_pixmap(rgb)
            s_nm = np.asarray(box.get("strip_s_nm", []), dtype=np.float64)
            offsets_nm = np.asarray(box.get("strip_offsets_nm", []), dtype=np.float64)
            s_extent = abs(float(s_nm[-1]) - float(s_nm[0])) if s_nm.size >= 2 else max(length_nm, 1.0)
            offset_extent = abs(float(offsets_nm[-1]) - float(offsets_nm[0])) if offsets_nm.size >= 2 else 2.0 * float(box.get("box_half_width_nm", 10.0))
            scale = max(float(nm_per_px), 1e-9)
            image_width = max(1, int(round(s_extent / scale)))
            image_height = max(8, int(round(offset_extent / scale)))
            pixmap = pixmap.scaled(image_width, image_height, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation)
            image_label.setPixmap(pixmap)
            image_label.setFixedSize(image_width, image_height)
        else:
            image_label.setText(self._tr("empty"))
            image_label.setFixedSize(text_width, image_height)
        layout.addWidget(image_label)

        profile_label = QtWidgets.QLabel()
        profile_label.setPixmap(
            _profile_to_pixmap(
                np.asarray(box.get("strip_s_nm", []), dtype=np.float64),
                np.asarray(box.get("height_profile_nm", []), dtype=np.float64),
                np.asarray(box.get("fwhm_profile_nm", []), dtype=np.float64),
                image_width,
            )
        )
        profile_label.setFixedSize(image_width, 46)
        profile_label.setToolTip(
            "blue: height above background, orange: FWHM"
            if self._ui_language == UI_LANG_EN else
            "青: 背景差し引き高さ、橙: FWHM"
        )
        layout.addWidget(profile_label)

        plane_text = self._tr("on") if bool(box.get("plane_corrected", False)) else self._tr("off")
        meta = QtWidgets.QLabel(
            f"{self._tr('box half width')} {float(box.get('box_half_width_nm', float('nan'))):.1f} nm, "
            f"{self._tr('plane correction')} {plane_text}, "
            f"{self._tr('profile points')} {np.sum(np.isfinite(np.asarray(box.get('height_profile_nm', []), dtype=np.float64)))}"
        )
        meta.setWordWrap(True)
        meta.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(meta)
        widget.setMinimumWidth(max(text_width, image_width))
        return widget

    def _export_filament_boxes(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        prefix = self._get_current_image_id()
        self._export_filament_box_summary_csv(os.path.join(out_dir, f"{prefix}_filament_box_summary.csv"))
        self._export_filament_box_profiles_csv(os.path.join(out_dir, f"{prefix}_filament_box_profiles.csv"))
        for box in self._stored_filament_boxes:
            box_id = int(box.get("box_id", 0))
            base = os.path.join(out_dir, f"{prefix}_box_{box_id:03d}_strip")
            strip = np.asarray(box.get("strip_image", []), dtype=np.float64)
            np.savetxt(base + ".csv", strip, delimiter=",", fmt="%.8g")
            np.save(base + ".npy", strip)
            self._export_filament_box_axes_csv(base + "_axes.csv", box)
            self._export_filament_box_png(base + ".png", box)

    def _export_filament_box_summary_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "box_id", "source_path", "image_id", "frame", "fiber_group_id",
                "length_nm", "box_half_width_nm", "plane_corrected", "mean_height_nm",
                "median_height_nm", "max_height_nm", "mean_fwhm_nm", "median_fwhm_nm",
                "valid_profile_points",
            ])
            for box in self._stored_filament_boxes:
                heights = np.asarray(box.get("height_profile_nm", []), dtype=np.float64)
                widths = np.asarray(box.get("fwhm_profile_nm", []), dtype=np.float64)
                h_ok = heights[np.isfinite(heights)]
                w_ok = widths[np.isfinite(widths)]
                writer.writerow([
                    box.get("box_id", ""),
                    box.get("source_path", ""),
                    box.get("image_id", ""),
                    int(box.get("frame_index", 0)) + 1,
                    box.get("fiber_group_id", ""),
                    self._csv_number(box.get("length_nm")),
                    self._csv_number(box.get("box_half_width_nm")),
                    "1" if bool(box.get("plane_corrected", False)) else "0",
                    self._csv_number(float(np.mean(h_ok)) if h_ok.size else np.nan),
                    self._csv_number(float(np.median(h_ok)) if h_ok.size else np.nan),
                    self._csv_number(float(np.max(h_ok)) if h_ok.size else np.nan),
                    self._csv_number(float(np.mean(w_ok)) if w_ok.size else np.nan),
                    self._csv_number(float(np.median(w_ok)) if w_ok.size else np.nan),
                    int(h_ok.size),
                ])

    def _export_filament_box_profiles_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "box_id", "source_path", "image_id", "frame", "plane_corrected", "s_nm",
                "height_above_bg_nm", "fwhm_nm", "background_nm", "peak_offset_nm",
            ])
            for box in self._stored_filament_boxes:
                s_nm = np.asarray(box.get("strip_s_nm", []), dtype=np.float64)
                height = np.asarray(box.get("height_profile_nm", []), dtype=np.float64)
                fwhm = np.asarray(box.get("fwhm_profile_nm", []), dtype=np.float64)
                bg = np.asarray(box.get("background_profile_nm", []), dtype=np.float64)
                peak = np.asarray(box.get("peak_offset_nm", []), dtype=np.float64)
                n = min(s_nm.size, height.size, fwhm.size, bg.size, peak.size)
                for idx in range(n):
                    writer.writerow([
                        box.get("box_id", ""),
                        box.get("source_path", ""),
                        box.get("image_id", ""),
                        int(box.get("frame_index", 0)) + 1,
                        "1" if bool(box.get("plane_corrected", False)) else "0",
                        self._csv_number(s_nm[idx]),
                        self._csv_number(height[idx]),
                        self._csv_number(fwhm[idx]),
                        self._csv_number(bg[idx]),
                        self._csv_number(peak[idx]),
                    ])

    def _export_filament_box_axes_csv(self, path: str, box: Dict[str, Any]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["axis", "index", "value_nm"])
            for idx, value in enumerate(np.asarray(box.get("strip_s_nm", []), dtype=np.float64)):
                writer.writerow(["s", idx, self._csv_number(value)])
            for idx, value in enumerate(np.asarray(box.get("strip_offsets_nm", []), dtype=np.float64)):
                writer.writerow(["offset", idx, self._csv_number(value)])

    def _export_filament_box_png(self, path: str, box: Dict[str, Any]) -> None:
        strip = np.asarray(box.get("strip_image", []), dtype=np.float64)
        finite = strip[np.isfinite(strip)]
        if finite.size:
            vmin = float(np.percentile(finite, 1.0))
            vmax = float(np.percentile(finite, 99.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        rgb = _strip_to_rgb(strip, vmin, vmax)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        image.copy().save(path)

    @staticmethod
    def _csv_number(value: Any) -> str:
        try:
            v = float(value)
        except Exception:
            return ""
        if not np.isfinite(v):
            return ""
        return f"{v:.8g}"

    def _filament_box_to_payload(self, box: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "box_id": int(box.get("box_id", 0)),
            "source_path": box.get("source_path", ""),
            "image_id": box.get("image_id", ""),
            "frame_index": int(box.get("frame_index", 0)),
            "fiber_group_id": box.get("fiber_group_id"),
            "length_nm": box.get("length_nm"),
            "box_half_width_nm": box.get("box_half_width_nm"),
            "plane_corrected": bool(box.get("plane_corrected", False)),
            "path_full_xy": box.get("path_full_xy", []),
            "points_yx": _array_to_json_list(box.get("points_yx", [])),
            "strip_image": _array_to_json_list(box.get("strip_image", [])),
            "strip_s_nm": _array_to_json_list(box.get("strip_s_nm", [])),
            "strip_offsets_nm": _array_to_json_list(box.get("strip_offsets_nm", [])),
            "height_profile_nm": _array_to_json_list(box.get("height_profile_nm", [])),
            "fwhm_profile_nm": _array_to_json_list(box.get("fwhm_profile_nm", [])),
            "background_profile_nm": _array_to_json_list(box.get("background_profile_nm", [])),
            "peak_offset_nm": _array_to_json_list(box.get("peak_offset_nm", [])),
        }

    def _filament_box_from_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        strip = _array_from_payload(payload.get("strip_image"), 2, dtype=float)
        strip_s = _array_from_payload(payload.get("strip_s_nm"), 1, dtype=float)
        offsets = _array_from_payload(payload.get("strip_offsets_nm"), 1, dtype=float)
        if strip.size == 0:
            return None
        profiles = compute_filament_box_profiles(strip, offsets)
        height = _array_from_payload(payload.get("height_profile_nm"), 1, dtype=float)
        fwhm = _array_from_payload(payload.get("fwhm_profile_nm"), 1, dtype=float)
        bg = _array_from_payload(payload.get("background_profile_nm"), 1, dtype=float)
        peak = _array_from_payload(payload.get("peak_offset_nm"), 1, dtype=float)
        if height.size != strip.shape[1]:
            height = profiles["height_nm"]
        if fwhm.size != strip.shape[1]:
            fwhm = profiles["fwhm_nm"]
        if bg.size != strip.shape[1]:
            bg = profiles["background_nm"]
        if peak.size != strip.shape[1]:
            peak = profiles["peak_offset_nm"]
        try:
            box_id = int(payload.get("box_id", 0))
        except Exception:
            box_id = 0
        if box_id <= 0:
            box_id = self._next_filament_box_id
        return {
            "box_id": box_id,
            "source_path": payload.get("source_path", ""),
            "image_id": payload.get("image_id", ""),
            "frame_index": int(payload.get("frame_index", 0)),
            "fiber_group_id": payload.get("fiber_group_id"),
            "length_nm": payload.get("length_nm"),
            "box_half_width_nm": payload.get("box_half_width_nm"),
            "plane_corrected": bool(payload.get("plane_corrected", False)),
            "path_full_xy": payload.get("path_full_xy", []),
            "points_yx": _array_from_payload(payload.get("points_yx"), 2, dtype=float),
            "strip_image": strip,
            "strip_s_nm": strip_s,
            "strip_offsets_nm": offsets,
            "height_profile_nm": height,
            "fwhm_profile_nm": fwhm,
            "background_profile_nm": bg,
            "peak_offset_nm": peak,
        }

    def _fluctuation_result_to_payload(self, result: dict) -> Dict[str, Any]:
        array_keys = [
            "s_nm", "kappa_all", "kappa_mean", "kappa_var", "kappa_var_se", "lp_local_nm",
            "times_s", "frame_indices", "autocorr", "autocorr_dt_s", "autocorr_fit",
        ]
        payload: Dict[str, Any] = {}
        for key in array_keys:
            value = result.get(key)
            if value is not None:
                payload[key] = _array_to_json_list(value)
        for key in ("fit_amp", "tau_s", "lp_nm", "L_min_nm", "valid_points", "warning"):
            payload[key] = result.get(key)
        return payload

    def _fluctuation_result_from_payload(self, payload: Dict[str, Any]) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None
        result = {
            "s_nm": _array_from_payload(payload.get("s_nm"), 1, dtype=float),
            "kappa_all": _array_from_payload(payload.get("kappa_all"), 2, dtype=float),
            "kappa_mean": _array_from_payload(payload.get("kappa_mean"), 1, dtype=float),
            "kappa_var": _array_from_payload(payload.get("kappa_var"), 1, dtype=float),
            "kappa_var_se": _array_from_payload(payload.get("kappa_var_se"), 1, dtype=float),
            "lp_local_nm": _array_from_payload(payload.get("lp_local_nm"), 1, dtype=float),
            "times_s": _array_from_payload(payload.get("times_s"), 1, dtype=float),
            "frame_indices": _array_from_payload(payload.get("frame_indices"), 1, dtype=int),
            "autocorr": _array_from_payload(payload.get("autocorr"), 1, dtype=float),
            "autocorr_dt_s": _array_from_payload(payload.get("autocorr_dt_s"), 1, dtype=float),
        }
        fit = payload.get("autocorr_fit")
        result["autocorr_fit"] = None if fit is None else _array_from_payload(fit, 1, dtype=float)
        for key in ("fit_amp", "tau_s", "lp_nm", "L_min_nm", "valid_points", "warning"):
            result[key] = payload.get(key)
        if result["s_nm"].size == 0 or result["kappa_all"].size == 0:
            return None
        if result["lp_local_nm"].size != result["kappa_var"].size:
            result["lp_local_nm"] = _local_persistence_length_from_variance(
                result["s_nm"], result["kappa_var"]
            )
        return result

    @staticmethod
    def _record_file_token(file_id: Any) -> Optional[str]:
        if file_id is None or file_id == "":
            return None
        text = str(file_id)
        if os.path.isabs(text) or os.sep in text or "/" in text or "\\" in text:
            try:
                return os.path.normcase(os.path.normpath(os.path.abspath(text)))
            except Exception:
                return os.path.normcase(os.path.normpath(text))
        return text

    @staticmethod
    def _record_group_token(group_id: Any) -> Any:
        if group_id is None or group_id == "":
            return None
        try:
            return int(group_id)
        except Exception:
            return str(group_id)

    def _record_identity_key(self, rec: Dict[str, Any]) -> Tuple[Optional[str], int, Any]:
        try:
            frame_index = int(rec.get("frame_index", 0))
        except Exception:
            frame_index = 0
        return (
            self._record_file_token(rec.get("file_id")),
            frame_index,
            self._record_group_token(rec.get("fiber_group_id")),
        )

    def _upsert_recorded_contour(self, rec: Dict[str, Any]) -> bool:
        """
        Add or replace a contour record.
        Returns True when an existing same file/frame/group record was replaced.
        """
        target_key = self._record_identity_key(rec)
        matches = [
            idx for idx, old_rec in enumerate(self._recorded_contours_list)
            if self._record_identity_key(old_rec) == target_key
        ]
        if not matches:
            self._recorded_contours_list.append(rec)
            return False
        first = matches[0]
        self._recorded_contours_list[first] = rec
        for idx in reversed(matches[1:]):
            self._recorded_contours_list.pop(idx)
        return True

    def _on_ok_record(
        self,
        assign_current_group: Optional[bool] = None,
    ) -> None:
        """Record current contour length and path; draw the path on Full AFM image."""
        if self.full_viz_window is not None:
            path_pixels = getattr(self.full_viz_window, "_path_pixels", None)
            endpoints = getattr(self.full_viz_window, "_endpoints", None) or []
            if (not path_pixels or len(path_pixels) < 2) and len(endpoints) >= 2:
                self._rerun_trace_if_ready()
        if self.full_viz_window is None or not self.full_viz_window._path_pixels or len(self.full_viz_window._path_pixels) < 2:
            title = "Record" if self._ui_language == UI_LANG_EN else "記録"
            QtWidgets.QMessageBox.information(self, title, self._tr("輪郭を計測してからOKを押してください。"))
            return
        if self.manual_roi is None:
            return
        path_pixels = self.full_viz_window._path_pixels
        length_px = self.full_viz_window._path_length
        if length_px is None:
            return
        x0 = float(self.manual_roi.get("x0", 0))
        y0 = float(self.manual_roi.get("y0", 0))
        path_full_xy = [(x0 + p[1], y0 + p[0]) for p in path_pixels]
        nm_per_px = self._get_nm_per_pixel()
        length_nm = (length_px * nm_per_px) if (nm_per_px and nm_per_px > 0) else None
        end_to_end_px = self._compute_current_end_to_end_px()
        analysis_mode = getattr(self.full_viz_window, "_analysis_mode", self._get_analysis_mode())
        point_count_raw = getattr(self.full_viz_window, "_anchor_point_count", None)
        point_count = int(point_count_raw) if point_count_raw is not None else len(getattr(self.full_viz_window, "_endpoints", []) or [])
        persistence_px: Optional[float] = None
        persistence_nm: Optional[float] = None
        persistence_source = "path"
        if self.persistence_check.isChecked():
            persistence_px, persistence_source = self._compute_current_persistence_px(analysis_mode)
            if persistence_px is not None and nm_per_px and nm_per_px > 0:
                persistence_nm = persistence_px * nm_per_px
        fi = self._get_current_frame_index()
        file_id = self._get_current_file_id()
        if assign_current_group is None:
            assign_current_group = bool(
                getattr(self, "ok_add_group_check", None) is not None
                and self.ok_add_group_check.isChecked()
            )
        record_group_id = self._current_fiber_group_id if assign_current_group else None
        rec = {
            "path_full_xy": path_full_xy,
            "length_px": length_px,
            "length_nm": length_nm,
            "end_to_end_px": end_to_end_px,
            "analysis_mode": analysis_mode,
            "point_count": point_count,
            "persistence_length_px": persistence_px,
            "persistence_length_nm": persistence_nm,
            "persistence_source": persistence_source,
            "frame_index": fi,
            "file_id": file_id,
            "fiber_group_id": record_group_id,
        }
        self._upsert_recorded_contour(rec)
        self._last_ok_trace_signature = self._current_trace_signature()
        self._update_recorded_table()
        self._refresh_view()

    def _on_data_clear(self) -> None:
        """Clear all recorded contour data, table, and AFM overlay."""
        self._recorded_contours_list.clear()
        self._update_recorded_table()
        self._refresh_view()

    def _on_clear_trace_points(self) -> None:
        """Clear clicked trace points and any calculated contour."""
        if self.full_viz_window is not None:
            self.full_viz_window.clear_contour()
        self._reset_current_trace_labels()
        self._refresh_view()

    def _reset_current_trace_labels(self) -> None:
        """Reset contour result labels while keeping any clicked point count/end-to-end display."""
        self.length_label.setText(f"{self._tr('輪郭長')}: — nm")
        point_count = 0
        if self.full_viz_window is not None:
            point_count = len(getattr(self.full_viz_window, "_endpoints", []) or [])
        if point_count > 0:
            self.point_count_label.setText(f"{self._tr('指定点数')}: {point_count}")
        else:
            self.point_count_label.setText(f"{self._tr('点数')}: —")
        end_to_end_px = self._compute_current_end_to_end_px()
        nm_per_px = self._get_nm_per_pixel()
        if end_to_end_px is not None and nm_per_px is not None and nm_per_px > 0:
            self.end_to_end_label.setText(f"End-to-End: {end_to_end_px * nm_per_px:.3f} nm")
        elif end_to_end_px is not None:
            self.end_to_end_label.setText(f"End-to-End: {end_to_end_px:.3f} px ({self._tr('スケール不明')})")
        else:
            self.end_to_end_label.setText("End-to-End: —")
        self.persistence_label.setText(
            f"Persistence Length: {self._tr('計算待ち')}"
            if self.persistence_check.isChecked() and point_count >= 2 else
            "Persistence Length: —"
        )

    def _default_session_path(self) -> str:
        """Default save path: same dir and base name as first measured file, extension .contour_session.json."""
        if not self._recorded_contours_list:
            return os.path.join(os.getcwd(), "contour_session.json")
        first_file_id = self._recorded_contours_list[0].get("file_id") or ""
        if isinstance(first_file_id, str) and (os.sep in first_file_id or "/" in first_file_id):
            base = os.path.splitext(os.path.basename(first_file_id))[0]
            dirpath = os.path.dirname(first_file_id)
            if dirpath:
                return os.path.join(dirpath, base + ".contour_session.json")
        return os.path.join(os.getcwd(), "contour_session.json")

    def _current_afm_data_dir(self) -> str:
        current_file_id = self._get_current_file_id()
        if current_file_id:
            dirpath = os.path.dirname(os.path.abspath(str(current_file_id)))
            if os.path.isdir(dirpath):
                return dirpath
        for attr in ("lastOpenedDir", "loadDir"):
            dirpath = getattr(gv, attr, "")
            if isinstance(dirpath, str) and dirpath and os.path.isdir(dirpath):
                return dirpath
        return os.getcwd()

    @staticmethod
    def _safe_output_folder_name(name: str) -> str:
        text = str(name or "").strip()
        safe = []
        for ch in text:
            if ch in {os.sep, "/", "\\", ":", "\0"} or ord(ch) < 32:
                safe.append("_")
            else:
                safe.append(ch)
        folder_name = "".join(safe).strip(" .")
        return folder_name or "filament_analysis"

    def _default_fluctuation_csv_dir(self) -> str:
        file_id = self._get_current_file_id()
        if file_id:
            folder_name = os.path.splitext(os.path.basename(str(file_id)))[0]
        else:
            folder_name = self._get_current_image_id()
        folder_name = self._safe_output_folder_name(folder_name)
        return os.path.join(self._current_afm_data_dir(), folder_name)

    def _default_linearization_export_dir(self) -> str:
        return os.path.join(self._default_fluctuation_csv_dir(), "Linearization")

    def _default_recorded_measurements_csv_path(self) -> str:
        file_id = self._get_current_file_id()
        if file_id:
            stem = os.path.splitext(os.path.basename(str(file_id)))[0]
        else:
            stem = self._get_current_image_id()
        stem = self._safe_output_folder_name(stem)
        return os.path.join(self._current_afm_data_dir(), f"{stem}_filament_measurements.csv")

    def _save_session(self) -> None:
        """Save session (recorded contours, ROI, file info) to JSON. Default path from first measured file; overwrite prompt if exists."""
        file_paths = list(getattr(gv, "files", []) or [])
        try:
            file_paths = [str(p) for p in file_paths]
        except Exception:
            file_paths = []
        self._session_file_paths = list(file_paths)
        current_file_index = int(getattr(gv, "currentFileNum", 0))
        if current_file_index < 0 or current_file_index >= len(file_paths):
            current_file_index = 0
        current_frame_index = int(getattr(gv, "index", 0))
        roi_by_frame_str = {str(k): v for k, v in self.roi_by_frame.items()}
        payload = {
            "version": 1,
            "file_paths": file_paths,
            "current_file_index": current_file_index,
            "current_frame_index": current_frame_index,
            "manual_roi": self.manual_roi,
            "roi_by_frame": roi_by_frame_str,
            "recorded_contours": self._recorded_contours_list,
            "current_fiber_group_id": self._current_fiber_group_id,
            "next_fiber_group_id": self._next_fiber_group_id,
            "ui_settings": {
                "fps": float(self.fps_spin.value()) if hasattr(self, "fps_spin") else 0.5,
                "n_resample": int(self.n_resample_spin.value()) if hasattr(self, "n_resample_spin") else 100,
                "box_half_width_nm": float(self.box_half_width_spin.value()) if hasattr(self, "box_half_width_spin") else 10.0,
                "box_plane_correct": bool(self.box_plane_correct_check.isChecked()) if hasattr(self, "box_plane_correct_check") else False,
                "ok_add_group": bool(self.ok_add_group_check.isChecked()) if hasattr(self, "ok_add_group_check") else True,
                "ui_language": self._ui_language,
            },
            "stored_filament_boxes": [
                self._filament_box_to_payload(box) for box in self._stored_filament_boxes
            ],
            "next_filament_box_id": self._next_filament_box_id,
            "last_fluctuation_results": {
                str(group_id): self._fluctuation_result_to_payload(result)
                for group_id, result in self._last_fluctuation_results_by_group.items()
            },
        }
        default_path = self._default_session_path()
        default_dir = os.path.dirname(default_path)
        default_name = os.path.basename(default_path)
        path = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session",
            default_path,
            "JSON (*.json);;All Files (*)",
        )[0]
        if not path:
            return
        path = path.strip()
        if not path:
            return
        if not path.endswith(".json"):
            path = path + ".json"
        while os.path.exists(path):
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle(self._tr("上書き確認"))
            if self._ui_language == UI_LANG_EN:
                msg.setText(f"File '{os.path.basename(path)}' already exists. Overwrite it?")
            else:
                msg.setText(f"ファイル '{os.path.basename(path)}' は既に存在します。上書きしますか？")
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            msg.setButtonText(QtWidgets.QMessageBox.Yes, self._tr("上書き"))
            msg.setButtonText(QtWidgets.QMessageBox.No, self._tr("ファイル名を変更"))
            ret = msg.exec_()
            if ret == QtWidgets.QMessageBox.Yes:
                break
            path = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Session",
                default_path,
                "JSON (*.json);;All Files (*)",
            )[0]
            if not path or not path.strip():
                return
            path = path.strip()
            if not path.endswith(".json"):
                path = path + ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            QtWidgets.QMessageBox.information(self, "Save Session", f"{self._tr('セッションを保存しました')}:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Session", f"{self._tr('保存に失敗しました')}:\n{e}")

    def _load_session(self) -> None:
        """Load session from JSON and restore contours, ROI, and optionally file selection."""
        default_dir = self._current_afm_data_dir()
        path = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Session",
            default_dir,
            "JSON (*.json);;All Files (*)",
        )[0]
        if not path or not path.strip():
            return
        path = path.strip()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Session", f"{self._tr('読み込みに失敗しました')}:\n{e}")
            return
        version = data.get("version", 0)
        if version != 1:
            QtWidgets.QMessageBox.warning(self, "Load Session", f"{self._tr('不明なバージョンです')}: {version}")
            return
        self._recorded_contours_list = data.get("recorded_contours", [])
        try:
            self._next_fiber_group_id = int(data.get("next_fiber_group_id", 1))
        except Exception:
            self._next_fiber_group_id = 1
        max_group_id = 0
        for rec in self._recorded_contours_list:
            try:
                group_id = rec.get("fiber_group_id")
                if group_id is not None:
                    max_group_id = max(max_group_id, int(group_id))
            except Exception:
                continue
        if self._next_fiber_group_id <= max_group_id:
            self._next_fiber_group_id = max_group_id + 1
        current_group_raw = data.get("current_fiber_group_id")
        try:
            self._current_fiber_group_id = int(current_group_raw) if current_group_raw is not None else None
        except Exception:
            self._current_fiber_group_id = None
        if self._current_fiber_group_id is not None and self._next_fiber_group_id <= self._current_fiber_group_id:
            self._next_fiber_group_id = self._current_fiber_group_id + 1
        self._update_fiber_group_label()

        ui_settings = data.get("ui_settings", {}) or {}
        lang = ui_settings.get("ui_language")
        if lang in (UI_LANG_JA, UI_LANG_EN):
            self._ui_language = lang
        if hasattr(self, "fps_spin"):
            try:
                self.fps_spin.setValue(float(ui_settings.get("fps", self.fps_spin.value())))
            except Exception:
                pass
        if hasattr(self, "n_resample_spin"):
            try:
                self.n_resample_spin.setValue(int(ui_settings.get("n_resample", self.n_resample_spin.value())))
            except Exception:
                pass
        if hasattr(self, "box_half_width_spin"):
            try:
                self.box_half_width_spin.setValue(float(ui_settings.get("box_half_width_nm", self.box_half_width_spin.value())))
            except Exception:
                pass
        if hasattr(self, "box_plane_correct_check"):
            try:
                self.box_plane_correct_check.setChecked(bool(ui_settings.get("box_plane_correct", self.box_plane_correct_check.isChecked())))
            except Exception:
                pass
        if hasattr(self, "ok_add_group_check"):
            try:
                self.ok_add_group_check.setChecked(bool(ui_settings.get("ok_add_group", self.ok_add_group_check.isChecked())))
            except Exception:
                pass
        try:
            self._next_filament_box_id = int(data.get("next_filament_box_id", 1))
        except Exception:
            self._next_filament_box_id = 1
        loaded_boxes: List[Dict[str, Any]] = []
        for box_payload in data.get("stored_filament_boxes", []) or []:
            box = self._filament_box_from_payload(box_payload)
            if box is not None:
                loaded_boxes.append(box)
        self._stored_filament_boxes = loaded_boxes
        max_box_id = 0
        for box in self._stored_filament_boxes:
            try:
                max_box_id = max(max_box_id, int(box.get("box_id", 0)))
            except Exception:
                continue
        if self._next_filament_box_id <= max_box_id:
            self._next_filament_box_id = max_box_id + 1

        last_results: Dict[int, dict] = {}
        for group_id_text, result_payload in (data.get("last_fluctuation_results", {}) or {}).items():
            try:
                group_id = int(group_id_text)
            except Exception:
                continue
            result = self._fluctuation_result_from_payload(result_payload)
            if result is not None:
                last_results[group_id] = result
        self._last_fluctuation_results_by_group = last_results

        self.manual_roi = data.get("manual_roi")
        roi_by_frame_raw = data.get("roi_by_frame", {})
        try:
            self.roi_by_frame = {int(k): v for k, v in roi_by_frame_raw.items()}
        except (ValueError, TypeError):
            self.roi_by_frame = {}
        if self.manual_roi:
            self.roi_status_label.setText(self._tr("ROI選択済み"))
        else:
            self.roi_status_label.setText(self._tr("ROI未選択"))
        file_paths = data.get("file_paths", [])
        try:
            file_paths = [str(p) for p in file_paths]
        except Exception:
            file_paths = []
        self._session_file_paths = list(file_paths)
        current_file_index = int(data.get("current_file_index", 0))
        found_row = -1
        if file_paths and 0 <= current_file_index < len(file_paths) and self.main_window and hasattr(self.main_window, "FileList"):
            target_path = file_paths[current_file_index]
            try:
                target_path = str(target_path)
            except Exception:
                target_path = None
            if target_path:
                fl = self.main_window.FileList
                for row in range(fl.count()):
                    item = fl.item(row)
                    if item is None:
                        continue
                    try:
                        row_path = str(item.text() if hasattr(item, "text") else item.data(QtCore.Qt.UserRole) or "")
                        if row_path == target_path or target_path.endswith(row_path) or row_path.endswith(target_path):
                            found_row = row
                            break
                    except Exception:
                        continue
                if found_row >= 0:
                    try:
                        fl.setCurrentRow(found_row)
                        if hasattr(self.main_window, "ListClickFunction"):
                            try:
                                self.main_window.ListClickFunction(bring_to_front=False)
                            except TypeError:
                                self.main_window.ListClickFunction()
                    except Exception:
                        pass
        self._update_recorded_table()
        self._redraw_filament_box_catalog()
        self._retranslate_ui()
        if self._stored_filament_boxes:
            self._show_filament_box_catalog_window()
        if self._last_fluctuation_results_by_group:
            self._show_fluctuation_results_window(self._last_fluctuation_results_by_group)
            self.fluctuation_summary_label.setText(
                f"Restored previous analysis results for {len(self._last_fluctuation_results_by_group)} groups"
                if self._ui_language == UI_LANG_EN else
                f"{len(self._last_fluctuation_results_by_group)} グループの前回解析結果を復元"
            )
        else:
            self.fluctuation_summary_label.setText(self._tr("未実行"))
        self._refresh_view()
        if found_row >= 0:
            QtWidgets.QMessageBox.information(self, "Load Session", self._tr("セッションを読み込みました。"))
        elif file_paths:
            QtWidgets.QMessageBox.information(
                self, "Load Session", self._tr("セッションを読み込みました。同じファイルを開くとオーバーレイが表示されます。")
            )
        else:
            QtWidgets.QMessageBox.information(self, "Load Session", self._tr("セッションを読み込みました。"))

    def _update_recorded_table(self) -> None:
        """Fill the recorded contour table (all frames, ID continues across frames)."""
        rows = self._recorded_contours_list
        self.recorded_table.setRowCount(len(rows))
        for i, rec in enumerate(rows):
            self.recorded_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            mode = str(rec.get("analysis_mode") or ANALYSIS_MODE_DNA)
            self.recorded_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(self._analysis_mode_display_label(mode))
            )
            pt_count = rec.get("point_count")
            self.recorded_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(str(int(pt_count)) if pt_count is not None else "—")
            )
            ln = rec.get("length_nm")
            if ln is not None:
                self.recorded_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{ln:.3f}"))
            else:
                self.recorded_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{rec.get('length_px', 0):.3f}"))
            e2p = rec.get("end_to_end_px")
            if e2p is None:
                # Backward compatibility for old sessions without end-to-end fields.
                path_full_xy = rec.get("path_full_xy") or []
                if len(path_full_xy) >= 2:
                    try:
                        x0, y0 = path_full_xy[0]
                        x1, y1 = path_full_xy[-1]
                        e2p = float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
                    except Exception:
                        e2p = None
            if e2p is not None:
                self.recorded_table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{e2p:.3f}"))
            else:
                self.recorded_table.setItem(i, 4, QtWidgets.QTableWidgetItem("—"))
            pln = rec.get("persistence_length_nm")
            plp = rec.get("persistence_length_px")
            if pln is not None:
                self.recorded_table.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{pln:.3f} nm"))
            elif plp is not None:
                self.recorded_table.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{plp:.3f} px"))
            else:
                self.recorded_table.setItem(i, 5, QtWidgets.QTableWidgetItem("—"))
        self.recorded_table.scrollToBottom()
        self._update_fiber_group_info_label()

    def _copy_table_to_clipboard(self) -> None:
        """Copy table (selection or full) to clipboard as tab-separated for Excel."""
        table = self.recorded_table
        ranges = table.selectedRanges()
        if ranges:
            r = ranges[0]
            top, bottom = r.topRow(), r.bottomRow()
            left, right = r.leftColumn(), r.rightColumn()
            lines = []
            for row in range(top, bottom + 1):
                cells = []
                for col in range(left, right + 1):
                    it = table.item(row, col)
                    cells.append(it.text() if it is not None else "")
                lines.append("\t".join(cells))
            text = "\n".join(lines)
        else:
            header = "\t".join(
                table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else ""
                for c in range(table.columnCount())
            )
            lines = [header]
            for row in range(table.rowCount()):
                cells = []
                for col in range(table.columnCount()):
                    it = table.item(row, col)
                    cells.append(it.text() if it is not None else "")
                lines.append("\t".join(cells))
            text = "\n".join(lines)
        if text:
            cb = QtWidgets.QApplication.clipboard()
            cb.setText(text)

    def _on_save_recorded_measurements_csv(self) -> None:
        if not self._recorded_contours_list:
            QtWidgets.QMessageBox.information(self, self._tr("CSV Save"), self._tr("記録した測定値がありません。"))
            return
        default_path = self._default_recorded_measurements_csv_path()
        path = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "CSV Save",
            default_path,
            "CSV (*.csv);;All Files (*)",
        )[0]
        if not path or not path.strip():
            return
        path = path.strip()
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.recorded_table.horizontalHeaderItem(col).text()
                    if self.recorded_table.horizontalHeaderItem(col) is not None else ""
                    for col in range(self.recorded_table.columnCount())
                ])
                for row in range(self.recorded_table.rowCount()):
                    writer.writerow([
                        self.recorded_table.item(row, col).text()
                        if self.recorded_table.item(row, col) is not None else ""
                        for col in range(self.recorded_table.columnCount())
                    ])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, self._tr("CSV Save"), f"{self._tr('CSV保存に失敗しました')}:\n{exc}")
            return
        QtWidgets.QMessageBox.information(self, self._tr("CSV Save"), f"{self._tr('CSVを保存しました')}:\n{path}")

    def _on_recorded_table_context_menu(self, pos: QtCore.QPoint) -> None:
        """Right-click on table: show delete menu and remove the row and corresponding overlay."""
        index = self.recorded_table.indexAt(pos)
        row = index.row()
        if row < 0:
            return
        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction(self._tr("この行を削除"))
        action = menu.exec_(self.recorded_table.viewport().mapToGlobal(pos))
        if action is delete_action:
            if 0 <= row < len(self._recorded_contours_list):
                self._recorded_contours_list.pop(row)
                self._update_recorded_table()
                self._refresh_view()

    def _on_full_image_selected(self, roi_info: Dict[str, Any]) -> None:
        w = roi_info.get("w", 0)
        h = roi_info.get("h", 0)
        if w <= 1 or h <= 1:
            return
        fi = self._get_current_frame_index()
        self.roi_by_frame[fi] = roi_info
        self.manual_roi = roi_info
        self.roi_status_label.setText(self._tr("ROI選択済み"))
        if self.full_viz_window:
            self.full_viz_window.clear_contour()
        self._refresh_view()

    def _get_current_frame_index(self) -> int:
        return int(getattr(gv, "index", 0))

    def _get_current_file_id(self) -> Optional[str]:
        """Return a string that identifies the current file (path or index). None if no file."""
        if not hasattr(gv, "files") or not gv.files:
            return None
        idx = getattr(gv, "currentFileNum", -1)
        if idx < 0 or idx >= len(gv.files):
            return None
        try:
            return str(gv.files[idx])
        except Exception:
            return str(idx)

    def _get_current_image_id(self) -> str:
        file_id = self._get_current_file_id()
        if file_id:
            base = os.path.splitext(os.path.basename(str(file_id)))[0]
            if base:
                return base
        return "filament_analysis"

    def _get_nm_per_pixel(self) -> Optional[float]:
        """Return nm per pixel (average of X and Y) from gv scan size, or None if unavailable."""
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize"):
            return None
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel"):
            return None
        if getattr(gv, "XScanSize", 0) == 0 or getattr(gv, "YScanSize", 0) == 0:
            return None
        if getattr(gv, "XPixel", 0) == 0 or getattr(gv, "YPixel", 0) == 0:
            return None
        nm_x = gv.XScanSize / gv.XPixel
        nm_y = gv.YScanSize / gv.YPixel
        return (float(nm_x) + float(nm_y)) / 2.0

    def _get_nm_per_pixel_xy(self) -> Optional[Tuple[float, float]]:
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize"):
            return None
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel"):
            return None
        if getattr(gv, "XScanSize", 0) == 0 or getattr(gv, "YScanSize", 0) == 0:
            return None
        if getattr(gv, "XPixel", 0) == 0 or getattr(gv, "YPixel", 0) == 0:
            return None
        nm_x = float(gv.XScanSize) / float(gv.XPixel)
        nm_y = float(gv.YScanSize) / float(gv.YPixel)
        if not np.isfinite(nm_x) or not np.isfinite(nm_y) or nm_x <= 0 or nm_y <= 0:
            return None
        return nm_x, nm_y

    def _ensure_selection_loaded(self) -> bool:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            QtWidgets.QMessageBox.warning(self, "No Selection", self._tr("FileListが見つかりません。"))
            return False
        selected = self.main_window.FileList.selectedIndexes()
        target_row = selected[0].row() if selected else self.main_window.FileList.currentRow()
        if target_row is None or target_row < 0:
            QtWidgets.QMessageBox.information(self, "Select File", self._tr("ファイルリストで対象を選択してください。"))
            return False
        try:
            blocker = QtCore.QSignalBlocker(self.main_window.FileList)
            self.main_window.FileList.setCurrentRow(target_row)
        except Exception:
            self.main_window.FileList.setCurrentRow(target_row)
        if hasattr(self.main_window, "ListClickFunction"):
            try:
                self.main_window.ListClickFunction(bring_to_front=False)
            except TypeError:
                self.main_window.ListClickFunction()
        return True

    def _prepare_frame(self) -> Optional[np.ndarray]:
        if not hasattr(gv, "files") or not gv.files:
            QtWidgets.QMessageBox.information(self, "No Files", self._tr("ファイルがロードされていません。"))
            return None
        if getattr(gv, "currentFileNum", -1) < 0 or gv.currentFileNum >= len(gv.files):
            return None
        try:
            LoadFrame(gv.files[gv.currentFileNum])
            InitializeAryDataFallback()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"{self._tr('フレーム読み込みに失敗')}:\n{exc}")
            return None
        if not hasattr(gv, "aryData") or gv.aryData is None:
            return None
        # メイン画面と同じ: leveling/filter 適用済みデータを優先（aryData_processed_1ch）
        data_for_display = (
            gv.aryData_processed_1ch
            if (hasattr(gv, "aryData_processed_1ch") and gv.aryData_processed_1ch is not None)
            else gv.aryData
        )
        frame = np.asarray(data_for_display, dtype=np.float64)
        if frame.ndim != 2:
            return None
        return frame

    def _roi_bounds(self, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        if self.manual_roi is None:
            return None
        h_img, w_img = frame_shape
        x0 = int(round(self.manual_roi["x0"]))
        y0 = int(round(self.manual_roi["y0"]))
        x1 = int(round(self.manual_roi["x0"] + self.manual_roi["w"]))
        y1 = int(round(self.manual_roi["y0"] + self.manual_roi["h"]))
        x0c = max(x0, 0)
        y0c = max(y0, 0)
        x1c = min(x1, w_img)
        y1c = min(y1, h_img)
        return (x0c, y0c, x1c - x0c, y1c - y0c)

    def _get_roi_crop(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        if self.last_frame is None or self.manual_roi is None:
            return None, None
        bounds = self._roi_bounds(self.last_frame.shape)
        if bounds is None:
            return None, None
        roi_crop, _ = _crop_rect(self.last_frame, bounds)
        return roi_crop, self.manual_roi

    def _apply_preprocess(self, roi: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply UI-selected preprocessing for measurement and display."""
        if roi is None or roi.size == 0:
            return roi
        median_k = 0
        if getattr(self, "noise_check", None) and self.noise_check.isChecked() and getattr(self, "median_spin", None):
            median_k = int(self.median_spin.value())
        flatten_mode = "none"
        flatten_sigma = 10.0
        if getattr(self, "flatten_combo", None):
            mode_data = self.flatten_combo.currentData()
            if mode_data == "plane":
                flatten_mode = "plane"
            elif mode_data == "gaussian":
                flatten_mode = "gaussian"
                if getattr(self, "flatten_sigma_spin", None):
                    flatten_sigma = float(self.flatten_sigma_spin.value())
        contrast_mode = "none"
        contrast_low, contrast_high = 2.0, 98.0
        gamma = 1.0
        if getattr(self, "contrast_combo", None):
            contrast_data = self.contrast_combo.currentData()
            if contrast_data == "percentile":
                contrast_mode = "percentile"
                if getattr(self, "contrast_low_spin", None):
                    contrast_low = float(self.contrast_low_spin.value())
                if getattr(self, "contrast_high_spin", None):
                    contrast_high = float(self.contrast_high_spin.value())
            elif contrast_data == "gamma":
                contrast_mode = "gamma"
                if getattr(self, "gamma_spin", None):
                    gamma = float(self.gamma_spin.value())
        deconv_enable = (
            getattr(self, "deconv_check", None) is not None
            and self.deconv_check.isChecked()
        )
        deconv_sigma = 1.0
        deconv_iterations = 10
        if deconv_enable:
            if getattr(self, "deconv_sigma_spin", None):
                deconv_sigma = float(self.deconv_sigma_spin.value())
            if getattr(self, "deconv_iter_spin", None):
                deconv_iterations = int(self.deconv_iter_spin.value())
        return apply_roi_preprocess(
            roi,
            median_k=median_k,
            deconv_enable=deconv_enable,
            deconv_sigma=deconv_sigma,
            deconv_iterations=deconv_iterations,
            flatten_mode=flatten_mode,
            flatten_sigma=flatten_sigma,
            contrast_mode=contrast_mode,
            contrast_low=contrast_low,
            contrast_high=contrast_high,
            gamma=gamma,
        )

    def _refresh_view(self) -> None:
        if self.full_viz_window is None:
            return
        if self.last_frame is None:
            return
        self.full_viz_window.set_nm_per_pixel(self._get_nm_per_pixel())
        roi_crop, roi_overlay = self._get_roi_crop()
        roi_display = self._apply_preprocess(roi_crop) if roi_crop is not None else None
        self.full_viz_window.update_view(
            self.last_frame,
            roi_overlay=roi_overlay,
            roi_image=roi_display if roi_display is not None else roi_crop,
            recorded_contours=self._recorded_contours_list,
            current_frame_index=self._get_current_frame_index(),
            current_file_id=self._get_current_file_id(),
        )

    def _rerun_trace_if_ready(self) -> None:
        """If ROI exists and full window has 2 endpoints, re-run contour trace and update display."""
        if self.full_viz_window is None or self.last_frame is None:
            return
        roi_crop, _ = self._get_roi_crop()
        if roi_crop is None:
            return
        roi_display = self._apply_preprocess(roi_crop)
        if roi_display is None:
            roi_display = roi_crop
        self.full_viz_window.run_trace_and_length(roi_display, self._on_length_computed)
        self._refresh_view()

    def _on_ridge_sigma_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_ridge_sigma(self.ridge_sigma_spin.value())
        self._rerun_trace_if_ready()

    def _on_max_bending_changed(self) -> None:
        if self.full_viz_window:
            v = self.max_bending_spin.value()
            self.full_viz_window.set_max_bending_angle_deg(v if v > 0 else None)
        self._rerun_trace_if_ready()

    def _on_ridge_weight_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_ridge_weight(self.ridge_weight_spin.value())
        self._rerun_trace_if_ready()

    def _on_ridge_floor_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_ridge_floor_threshold(self.ridge_floor_spin.value())
        self._rerun_trace_if_ready()

    def _on_marker_style_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_marker_size(self.marker_size_spin.value())
            marker = self.marker_shape_combo.currentData()
            self.full_viz_window.set_marker_shape(str(marker) if marker else "o")
        self._refresh_view()

    def _on_cmap_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_cmap(self.cmap_combo.currentText().strip() or "viridis")
        self._refresh_view()

    def _get_analysis_mode(self) -> str:
        mode = self.analysis_mode_combo.currentData()
        if mode == ANALYSIS_MODE_BEADS:
            return ANALYSIS_MODE_BEADS
        return ANALYSIS_MODE_DNA

    def _on_analysis_mode_changed(self) -> None:
        mode = self._get_analysis_mode()
        if hasattr(self, "ridge_group") and self.ridge_group is not None:
            self.ridge_group.setEnabled(mode == ANALYSIS_MODE_DNA)
        if self.full_viz_window:
            self.full_viz_window.set_analysis_mode(mode)
            self.full_viz_window.clear_contour()
        self.length_label.setText(f"{self._tr('輪郭長')}: — nm")
        self.point_count_label.setText(f"{self._tr('点数')}: —")
        self.end_to_end_label.setText("End-to-End: —")
        if self.persistence_check.isChecked():
            self.persistence_label.setText(f"Persistence Length: {self._tr('計算待ち')}")
        else:
            self.persistence_label.setText("Persistence Length: —")
        self._refresh_view()

    def _on_persistence_toggle_changed(self) -> None:
        if not self.persistence_check.isChecked():
            self.persistence_label.setText("Persistence Length: —")
            return
        if self.full_viz_window and self.full_viz_window._path_length is not None:
            self._on_length_computed(self.full_viz_window._path_length)
        else:
            self.persistence_label.setText("Persistence Length: —")

    def _on_length_computed(self, length_px: float) -> None:
        nm_per_px = self._get_nm_per_pixel()
        if nm_per_px is not None and nm_per_px > 0:
            self.length_label.setText(f"{self._tr('輪郭長')}: {length_px * nm_per_px:.3f} nm")
        else:
            self.length_label.setText(f"{self._tr('輪郭長')}: {length_px:.3f} px ({self._tr('スケール不明')})")
        point_count = None
        if self.full_viz_window is not None:
            point_count = self.full_viz_window._anchor_point_count
        mode = getattr(self.full_viz_window, "_analysis_mode", self._get_analysis_mode())
        if point_count is not None:
            if mode == ANALYSIS_MODE_BEADS:
                self.point_count_label.setText(f"{self._tr('点数')}: {int(point_count)}")
            else:
                self.point_count_label.setText(f"{self._tr('指定点数')}: {int(point_count)}")
        else:
            self.point_count_label.setText(f"{self._tr('点数')}: —")

        end_to_end_px = self._compute_current_end_to_end_px()
        if end_to_end_px is None:
            self.end_to_end_label.setText(f"End-to-End: {self._tr('計算不可')}")
        elif nm_per_px is not None and nm_per_px > 0:
            self.end_to_end_label.setText(f"End-to-End: {end_to_end_px * nm_per_px:.3f} nm")
        else:
            self.end_to_end_label.setText(f"End-to-End: {end_to_end_px:.3f} px ({self._tr('スケール不明')})")

        if not self.persistence_check.isChecked():
            self.persistence_label.setText("Persistence Length: —")
            return
        lp_px, _src = self._compute_current_persistence_px(mode)
        if lp_px is None:
            self.persistence_label.setText(f"Persistence Length: {self._tr('計算不可')}")
            return
        if nm_per_px is not None and nm_per_px > 0:
            self.persistence_label.setText(f"Persistence Length: {lp_px * nm_per_px:.3f} nm")
        else:
            self.persistence_label.setText(f"Persistence Length: {lp_px:.3f} px ({self._tr('スケール不明')})")

    def _get_current_anchor_points(self) -> Optional[List[Tuple[float, float]]]:
        if self.full_viz_window is None:
            return None
        pts = self.full_viz_window._anchor_points_for_trace
        if pts and len(pts) >= 2:
            return pts
        eps = getattr(self.full_viz_window, "_endpoints", None)
        if eps and len(eps) >= 2:
            return [(float(p[0]), float(p[1])) for p in eps]
        return None

    def _compute_current_end_to_end_px(self) -> Optional[float]:
        points = self._get_current_anchor_points()
        if points and len(points) >= 2:
            return end_to_end_distance_from_points([points[0], points[-1]])
        if self.full_viz_window is None:
            return None
        path_pixels = getattr(self.full_viz_window, "_path_pixels", None)
        if path_pixels and len(path_pixels) >= 2:
            y0, x0 = path_pixels[0]
            y1, x1 = path_pixels[-1]
            return float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
        return None

    def _compute_current_persistence_px(self, analysis_mode: str) -> Tuple[Optional[float], str]:
        if analysis_mode == ANALYSIS_MODE_BEADS:
            pts = self._get_current_anchor_points() or []
            return persistence_length_2d_from_segments(pts), "segment"
        return persistence_length_2d_from_path(self._get_persistence_path_pixels() or []), "path"

    def _get_persistence_path_pixels(self) -> Optional[List[Tuple[int, int]]]:
        if self.full_viz_window is None:
            return None
        p = self.full_viz_window._persistence_path_pixels
        if p and len(p) >= 2:
            return p
        return self.full_viz_window._path_pixels

    def _connect_frame_signal(self) -> None:
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_frame_changed)
            except Exception:
                pass

    def _on_frame_changed(self, frame_index: int) -> None:
        frame_index_int = int(frame_index)
        previous_frame = self._last_frame_index_seen
        frame_changed = previous_frame is not None and frame_index_int != previous_frame
        if frame_changed and self.full_viz_window is not None:
            self.full_viz_window.clear_trace_result_keep_points()
            self._reset_current_trace_labels()

        pending_roi = self._pending_frame_navigation_roi
        if pending_roi is not None:
            self.roi_by_frame[frame_index_int] = dict(pending_roi)
            self.manual_roi = dict(pending_roi)
            self._pending_frame_navigation_roi = None
        elif frame_index_int not in self.roi_by_frame and frame_changed and self.manual_roi is not None:
            self.roi_by_frame[frame_index_int] = dict(self.manual_roi)
            self.manual_roi = self.roi_by_frame.get(frame_index_int)
        else:
            self.manual_roi = self.roi_by_frame.get(frame_index_int)
        if self.manual_roi is None:
            self.roi_status_label.setText(self._tr("ROI未選択"))
        else:
            self.roi_status_label.setText(self._tr("ROI選択済み"))
        frame = self._prepare_frame()
        if frame is None:
            return
        self.last_frame = frame
        self._refresh_view()
        self._update_frame_label()
        self._update_fiber_group_info_label()
        self._last_frame_index_seen = frame_index_int

    def _update_frame_label(self) -> None:
        total = int(getattr(gv, "FrameNum", 0)) or 0
        current = self._get_current_frame_index()
        if total > 0:
            self.frame_label.setText(f"{self._tr('Frame')}: {current + 1} / {total}")
        else:
            self.frame_label.setText(f"{self._tr('Frame')}: —")

    def _prev_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0)) or 0
        if total <= 0:
            return
        idx = max(0, self._get_current_frame_index() - 1)
        self._pending_frame_navigation_roi = dict(self.manual_roi) if self.manual_roi is not None else None
        self._set_frame_index(idx)

    def _next_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0)) or 0
        if total <= 0:
            return
        if not self._confirm_next_with_unrecorded_trace():
            return
        idx = min(total - 1, self._get_current_frame_index() + 1)
        self._pending_frame_navigation_roi = dict(self.manual_roi) if self.manual_roi is not None else None
        self._set_frame_index(idx)

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
        self._update_frame_label()

    def _show_full_image_view(self) -> None:
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            QtWidgets.QMessageBox.information(self, "No Data", self._tr("画像がロードされていません。"))
            return
        self.last_frame = frame
        if self.full_viz_window is None:
            self.full_viz_window = ContourLengthFullImageWindow(
                self,
                roi_basic_group=self.roi_basic_group,
                contour_length_group=self.contour_length_group,
                frame_nav_widget=self.frame_nav_widget,
            )
        self.full_viz_window.set_on_length_computed(self._on_length_computed)
        self.full_viz_window.set_ridge_sigma(self.ridge_sigma_spin.value())
        self.full_viz_window.set_ridge_weight(self.ridge_weight_spin.value())
        self.full_viz_window.set_ridge_floor_threshold(self.ridge_floor_spin.value())
        v = self.max_bending_spin.value()
        self.full_viz_window.set_max_bending_angle_deg(v if v > 0 else None)
        self.full_viz_window.set_analysis_mode(self._get_analysis_mode())
        self.full_viz_window.set_marker_size(self.marker_size_spin.value())
        marker = self.marker_shape_combo.currentData()
        self.full_viz_window.set_marker_shape(str(marker) if marker else "o")
        self.full_viz_window.set_cmap(self.cmap_combo.currentText().strip() or "viridis")
        self.full_viz_window.set_nm_per_pixel(self._get_nm_per_pixel())
        self.full_viz_window.enable_roi_selector("Rectangle", self._on_full_image_selected)
        roi_crop, roi_overlay = self._get_roi_crop()
        roi_display = self._apply_preprocess(roi_crop) if roi_crop is not None else None
        self.full_viz_window.update_view(
            frame,
            roi_overlay=roi_overlay,
            roi_image=roi_display if roi_display is not None else roi_crop,
            recorded_contours=self._recorded_contours_list,
            current_frame_index=self._get_current_frame_index(),
            current_file_id=self._get_current_file_id(),
        )
        if hasattr(self, "ridge_group") and self.ridge_group is not None:
            self.ridge_group.setEnabled(self._get_analysis_mode() == ANALYSIS_MODE_DNA)
        self.full_viz_window.show()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.full_viz_window is None:
            self._show_full_image_view()

    def closeEvent(self, event) -> None:
        if self.full_viz_window:
            self.full_viz_window.close()
            self.full_viz_window = None
        if self._fluctuation_results_window:
            self._fluctuation_results_window.close()
            self._fluctuation_results_window = None
        if self._filament_box_catalog_window:
            self._filament_box_catalog_window.close()
            self._filament_box_catalog_window = None
        super().closeEvent(event)


def _assert_curvature_straight_line() -> None:
    sample_path_pixels = [(0, i) for i in range(40)]
    xy = [(float(col), float(row)) for row, col in sample_path_pixels]
    xy_resampled, length = resample_path_uniform(xy, 40)
    ds = length / max(1, xy_resampled.shape[0] - 1)
    kappa = compute_curvature_from_path(xy_resampled, ds)
    finite = kappa[np.isfinite(kappa)]
    assert finite.size > 0
    assert float(np.max(np.abs(finite))) < 1e-9


_assert_curvature_straight_line()


def create_plugin(main_window):
    return ContourLengthWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin"]
