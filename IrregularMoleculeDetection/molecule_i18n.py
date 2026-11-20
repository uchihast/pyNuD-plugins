"""Japanese strings for the Irregular Molecule Detection plugin.
不定形分子の検出プラグインの日本語表示。

pyNuD's gui_i18n translates labels, buttons, check boxes, group titles, tooltips, combo
items and table headers through its catalogue (ui_translations.json in the pyNuD folder).
Installed pyNuD builds older than this plugin do not carry these strings, so the plugin
adds them to the catalogue when it is imported. Entries already in the catalogue win.
Text that the plugin builds at run time (summaries, messages) uses tr(). Plot text stays in English.
"""
from __future__ import annotations

STRINGS = {
    # --- panel
    "No ASD file loaded.": "ASDファイルが読み込まれていません。",
    "Analyze Current Frame": "現在のフレームを解析",
    "Analyze All Frames": "全フレーム解析",
    "Update on frame change": "フレーム変更時に更新",
    "Analyze the current frame again whenever the frame changes in pyNuD.":
        "pyNuD でフレームを変えるたびに、現在のフレームを解析し直します。",
    "Update on parameter change": "パラメータ変更時に更新",
    "Analyze the current frame again whenever a parameter on the left changes.":
        "左のパラメータを変えるたびに、現在のフレームを解析し直します。",
    "Export CSV": "CSVエクスポート",
    "Save Image": "画像を保存",
    "Restore Defaults": "既定値に戻す",
    "User Manual": "マニュアル",
    "How to use the plugin and what each parameter means.": "使い方と各パラメータの意味を表示します。",
    "Input": "入力",
    "Data:": "データ:",
    "Raw height": "元の高さ",
    "Processed image": "処理後の像",
    "Raw height: the frame as stored in the file (recommended; the plugin levels it itself).\n"
    "Processed image: the image after pyNuD's filters, if any were applied.":
        "元の高さ: ファイルに保存されたままのフレーム（推奨。傾き補正はプラグインが行う）。\n"
        "処理後の像: pyNuD のフィルターをかけた後の画像（かけていれば）。",
    "Background:": "背景:",
    "Polynomial (cropped image)": "多項式（切り出し像）",
    "Top-hat (raw wide field)": "トップハット（未処理の広視野）",
    "Polynomial: quadratic surface + scan-line median fitted outside the molecules.\n"
    "Use it for cropped, already flattened images; volumes of large clumps are preserved.\n"
    "Top-hat: removes structures wider than about twice the radius before detection.\n"
    "Use it for raw wide-field images with bumps or large particles; it also lowers the\n"
    "volume of clumps wider than about twice the radius.":
        "多項式: 分子の外側に 2 次曲面と走査線ごとの中央値を当てはめて引きます。\n"
        "切り出して平坦化済みの像に使います。大きな塊の体積も保たれます。\n"
        "トップハット: 半径の約 2 倍より幅の広い構造を、検出の前に取り除きます。\n"
        "凹凸や大きな粒子のある広視野の生データに使います。半径の約 2 倍より\n"
        "幅の広い塊では体積も小さくなります。",
    "Top-hat radius:": "トップハット半径:",
    "Top-hat radius. Structures wider than about twice this are treated as background.":
        "トップハットの半径。この約 2 倍より幅の広い構造は背景として扱います。",
    "Detection": "検出",
    "Strand scale min:": "鎖スケール下限:",
    "Strand scale max:": "鎖スケール上限:",
    "Smallest ridge-filter scale (about the strand half-width).":
        "リッジフィルターの最小スケール（鎖の半幅程度）。",
    "Largest ridge-filter scale.": "リッジフィルターの最大スケール。",
    "Low threshold:": "低い閾値:",
    "High threshold:": "高い閾値:",
    "Low threshold. Pixels above this are kept only if connected to a strong region.\n"
    "Lower values extend strands further but add noise.":
        "低い閾値。これを超える画素は、強い領域につながっている場合だけ残します。\n"
        "下げると鎖を先まで拾えますが、ノイズも増えます。",
    "High threshold. Regions must contain pixels above this.":
        "高い閾値。領域はこれを超える画素を含む必要があります。",
    "Core threshold:": "芯の閾値:",
    "A molecule must contain a part whose smoothed height exceeds this.\n"
    "It removes faint noise blobs. Set 0 to keep every region above the thresholds.":
        "分子は、ぼかした高さがこれを超える部分を含む必要があります。\n"
        "弱いノイズの塊を除くための条件です。0 にすると、閾値を超えた領域をすべて残します。",
    "Min area:": "最小面積:",
    "Objects smaller than this are discarded.": "これより小さい物体は捨てます。",
    "Molecule": "分子",
    "Target:": "対象:",
    "Largest molecule": "最大の分子",
    "All molecules": "すべての分子",
    "Largest molecule: measure only the main clump (cropped single-molecule images).\n"
    "All molecules: measure every detected molecule.":
        "最大の分子: 主な塊だけを測ります（1 分子を切り出した像向け）。\n"
        "すべての分子: 検出した分子をすべて測ります。",
    "Grouping gap:": "まとめる距離:",
    "Detected pieces closer than this are treated as one molecule.":
        "これより近い断片は 1 つの分子として扱います。",
    "Cut gaps between strands": "鎖の間の隙間を除く",
    "Redraw the outline on the lightly filtered height so that substrate\n"
    "between strands is not counted as part of the molecule.":
        "軽くフィルターした高さで輪郭を引き直し、鎖の間の基板を\n"
        "分子に含めないようにします。",
    "Island gap:": "島の距離:",
    "Islands separated from the main body by at most this distance are candidates.":
        "本体からこの距離以内にある島を候補にします。",
    "Min island length:": "島の最小長さ:",
    "Candidate islands are kept only if their skeleton is at least this long\n"
    "(broken strands). Shorter stubs are dropped.":
        "候補の島は、骨格がこの長さ以上のとき（途切れた鎖）だけ残します。\n"
        "これより短い切れ端は捨てます。",
    "Exclude molecules touching the edge": "端に接する分子を除外",
    "Molecules cut by the scan edge or the drift-correction border are excluded,\n"
    "because their size and volume are incomplete.":
        "走査範囲の端やドリフト補正の境界で切れた分子は、\n"
        "大きさと体積が不完全なので除外します。",
    # --- view
    "Show outlines": "輪郭を表示",
    "Show IDs": "IDを表示",
    "Colour max:": "カラー上限:",
    "Upper limit of the colour scale.": "カラースケールの上限。",
    "Auto": "自動",
    "Set the colour range from the image: from -3σ to the height\n"
    "that 99.5 % of the molecule pixels stay below.":
        "カラースケールの範囲を像から決めます。下限は -3σ、上限は\n"
        "分子の画素の 99.5 % がそれより低くなる高さです。",
    "Press Analyze Current Frame.": "「現在のフレームを解析」を押してください。",
    # --- table
    "Area (nm²)": "面積 (nm²)",
    "Volume (nm³)": "体積 (nm³)",
    "Max height (nm)": "最大高さ (nm)",
    "Mean height (nm)": "平均高さ (nm)",
    "Skeleton length (nm)": "骨格の長さ (nm)",
    "Regions": "領域数",
    # --- progress dialog (pyNuD translates native dialogs through the catalogue)
    "Analyzing frames...": "フレームを解析しています...",
    "Cancel": "キャンセル",
}

# Text built at run time. Keys are format strings; tr() formats them after translating.
RUNTIME = {
    "Frame {frame} / {total}, {xp}x{yp} px, {xs:g}x{ys:g} nm ({dx:.3g}x{dy:.3g} nm/px)":
        "フレーム {frame} / {total}、{xp}x{yp} px、{xs:g}x{ys:g} nm（{dx:.3g}x{dy:.3g} nm/px）",
    "Non-square pixels: analyzed on {s:.3g} nm square pixels.":
        "画素が正方でないため、{s:.3g} nm の正方画素に補間して解析しました。",
    "Frame {frame}: {n} molecule(s), background σ = {sigma:.3f} nm":
        "フレーム {frame}: 分子 {n} 個、背景 σ = {sigma:.3f} nm",
    ", {n} excluded at the edge": "、端で除外 {n} 個",
    "All {frames} frames: {n} molecule(s)": "全 {frames} フレーム: 分子 {n} 個",
    ", volume mean {mean:.0f} nm³, SD {sd:.0f} nm³": "、体積の平均 {mean:.0f} nm³、標準偏差 {sd:.0f} nm³",
    "Analysis failed: {error}": "解析に失敗しました: {error}",
    "Analysis canceled.": "解析を中止しました。",
    "Saved: {path}": "保存しました: {path}",
    "Open an ASD file in pyNuD first.": "先に pyNuD で ASD ファイルを開いてください。",
    "Run an analysis first.": "先に解析を実行してください。",
    "No height data is loaded. Open an ASD file in pyNuD first.":
        "高さデータが読み込まれていません。先に pyNuD で ASD ファイルを開いてください。",
    "The scan size or pixel count of the current file is unknown.":
        "現在のファイルの走査範囲または画素数が分かりません。",
    "Export CSV": "CSVエクスポート",
    "Save Image": "画像を保存",
    "CSV files (*.csv)": "CSV ファイル (*.csv)",
}

try:
    import gui_i18n as _host
except ImportError:  # pyNuD without the language switch: English only.
    _host = None


def install() -> None:
    """Add the plugin's strings to pyNuD's catalogue without overriding existing entries."""
    catalogue = getattr(_host, "_strings", None)
    if not isinstance(catalogue, dict):
        return
    for en, ja in STRINGS.items():
        catalogue.setdefault(en, {"ja": ja, "en": en})


def language() -> str:
    getter = getattr(_host, "get_language", None)
    return getter() if callable(getter) else "en"


def tr(text: str, **values) -> str:
    """Translate run-time text into the current pyNuD language and fill in the values."""
    if language() == "ja":
        text = RUNTIME.get(text) or STRINGS.get(text) or text
    return text.format(**values) if values else text


def add_language_listener(callback) -> None:
    adder = getattr(_host, "add_language_listener", None)
    if callable(adder):
        adder(callback)
