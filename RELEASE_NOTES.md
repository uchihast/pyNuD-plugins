# Plugin ZIP bundles — 2026-11-14

## 更新内容 / Changes

公開中の13プラグインすべてを更新しました。All 13 public plugins are updated.

**全プラグイン共通 / All plugins**
- pyNuD 2.14.0の国旗スイッチ（JPN/ENG）に合わせて、画面と操作メッセージを日本語だけ、または英語だけで表示します。グラフの軸名は英語のままです。 / Screens and operation messages follow the JPN/ENG flag switch in pyNuD 2.14.0 and appear in one language only. Graph axis labels stay in English.
- ウィンドウを開き直すたびに位置がずれる問題を修正しました（Windows/macOS）。 / Windows no longer drift each time they are reopened (Windows/macOS).

**個別の修正 / Plugin-specific fixes**
- **Particle Tracking 2026.11.14.1**: Kalman追跡で終了した軌跡を結果に残し、サブピクセル補正を正しく適用します。LAP / Kalmanは有効なリンクの数を優先します。非正方ピクセルでの探索範囲を修正しました。 / Kalman tracking keeps finished tracks, sub-pixel refinement is applied correctly, LAP / Kalman keep the maximum number of valid links, and gating respects non-square pixels.
- **Filament Analysis 2026.11.14.1**: 本体の処理済みフレーム（背景補正・フィルタ後）を解析に使います。非正方ピクセルの長さ、階段状にならない輪郭長、校正値の記録を修正し、グループ未割り当ての記録を置き換えずに残します。経路探索を高速化しました。 / Uses the host's processed frames, corrects lengths for non-square pixels with staircase-free contour lengths, records the calibration, keeps unassigned records, and traces paths faster.
- **Kymograph 2026.11.14.1**: 1列あたりのnmを作成時に保持して距離換算と基準位置に使い、本体の処理済みフレームから作成します。 / Keeps nm per column for distance conversion and reference positions, and builds kymographs from the host's processed frames.
- **AFM Movie Editor 2026.11.14.1**: 本体の処理済みフレームを使い、メイン画面の状態を変更しません。 / Uses the host's processed frames without changing the main window state.
- **Spot Analysis 2026.11.14.1**: 本体の処理済みフレームを使います。CSV保存の失敗を一覧で示し、成功通知と区別します。 / Uses the host's processed frames; CSV save failures are listed and no longer reported as success.
- **L-AFM Analysis 2026.11.14.1**: 3D再構成で最上段の高さの検出を保持します。 / 3D reconstruction keeps detections at the top height level.
- **Particle Analysis 2026.11.14.1**: All Framesの結果に作成元ファイルを記録し、別のファイルの結果を使わないようにしました。 / All Frames results record their source file and are discarded for a different file.
- **Dwell Analysis 2026.11.14.1**: 検出とマーク処理の境界条件を修正しました。 / Detection and marking edge cases are corrected.
- **Particle Cluster Analysis 2026.11.14**: 楕円フィットでY方向の校正を保持します。 / Ellipse fits keep the Y calibration.
- **Simulator Bridge 2026.11.14.1**: pyNuD Simulatorとの通信部分を更新しました。複数のpyNuDからの配信で一時ファイルを消し合わず、窓を閉じた後に配信中へ戻りません。pyNuDの表示言語（JPN/ENG）をSimulatorへ送り、Bridgeでつながっている間はSimulatorもpyNuDと同じ言語で表示します（pyNuD Simulator 2.2.0以降の日英切り替えに対応）。 / Updated the link to pyNuD Simulator: multiple publishers no longer remove each other's files, and closing the window stops publishing for good. It also sends pyNuD's display language (JPN/ENG), so a connected pyNuD Simulator follows it (for the language switch in pyNuD Simulator 2.2.0 or later).
- **Interactive Drift Compensation 2026.11.14 / Reference Object Cropper 2026.11.14.1 / Normal Mode Analysis 2026.11.14**: 日英切り替えへの対応のみ。 / Language switch support only.

このリリースのプラグインには **pyNuD 2.14.0以降** が必要です。These plugin versions require **pyNuD 2.14.0 or later**.

## インストール / Installation
1. 本体を2.14.0以降へ更新し、pyNuDを再起動します。Update the host to 2.14.0 or later and restart pyNuD.
2. 更新するプラグインとそのAIレビュー画面を閉じます。Close the plugin and any of its AI review windows.
3. 各ZIPをダウンロードし、**Plugin → Import Plugin ZIP...** で読み込みます。Import each individual ZIP; extraction is not required. Existing imported copies are preserved in separate folders.
4. プラグインのウィンドウタイトルで版（2026.11.14 または 2026.11.14.1）を確認します。Confirm the version in the plugin window title.

## Verification
All 13 ZIPs were imported with the pyNuD 2.14.0 importer and every entry point loaded. ZIP contents match the files declared in each plugin.json, and SHA-256 digests are listed in SHA256SUMS.txt. The 1,639-test regression suite passes for the plugin source snapshot. These checks validate packaging and imports; they do not validate every scientific workflow.

---

# Plugin ZIP bundles — 2026-10-20

## 更新内容 / Changes
- **L-AFM Analysis 2026.10.20**: 検証付きASD保存と非有限値の拒否に対応。再構成した局在密度・強度は取得AFM高さではありません。Validated ASD export rejects nonfinite values; reconstructed density/intensity is not acquired AFM height.
- **Interactive Drift Compensation 2026.10.20** (初公開 / first public ZIP): 1ch・2chに同じシフトを適用し、2ch電圧と物理スキャン幅を保持して保存します。Applies the same shift to both channels and preserves channel-2 voltage and physical scan size. The Apply checkbox identifies both channels when present.
- **Reference Object Cropper 2026.10.20** (初公開 / first public ZIP): 取得方向・物理ピクセルサイズ・2ch換算係数を保持してCropを個別ASDに保存します。Saves each crop as ASD with acquisition orientation, physical pixel size and channel conversion preserved. Bruker height crops retain their height type; voltage crops record their fixed offset in the ASD comment.

この3つには **pyNuD 2.13.1以降** が必要です。These three versions require **pyNuD 2.13.1 or later**.

## インストール / Installation
1. 本体を2.13.1以降へ更新し、pyNuDを再起動します。Update the host and restart pyNuD.
2. 更新するプラグインとそのAIレビュー画面を閉じます。Close the plugin and any of its AI review windows.
3. 各ZIPをダウンロードし、**Plugin → Import Plugin ZIP...**で読み込みます。Import each individual ZIP; extraction is not required. Existing imported copies are preserved in separate folders.
4. プラグインのウィンドウタイトルで **2026.10.20** を確認します。Confirm the version in the plugin window title.

Interactive Drift: select the original file, then Track → Apply or Save .asd. Both channels use the same shift. Missing data in a declared second channel prevents saving.
Reference Object Cropper: Set Reference → Crop Current, or manual Crop → Save. Acquire crops again after changing the source file.
L-AFM: use the final result Save. ASD uses a zero minimum; nonfinite values stop saving. Existing 3D TIFF output is unchanged.

変更のない10プラグインは、旧公開ZIPをバイト単位でそのまま引き継ぎ、番号も変更していません。The other 10 public ZIPs and their versions are unchanged. This collection includes all 13 public plugins so latest-download links remain valid.

## Verification

71 related regression tests passed. All 13 ZIPs were imported with the packaged pyNuD 2.13.1 macOS runtime; entry points and dedicated modules loaded, and the three updated plugin windows showed version 2026.10.20. ZIP contents and SHA-256 digests were checked against the declared files. These checks validate packaging, imports and the covered save regressions; they do not validate every scientific workflow.

Plugin source snapshot: `4077336` (host 2.13.1; plugin runtime fixes through `1893bcf`).

---

# Plugin ZIP bundles — 2026-10-04

## Installation / インストール

Use **pyNuD 2.13.0 or later**. Download one plugin ZIP below and choose **Plugin → Import Plugin ZIP...**. Manual extraction is not required. For an already extracted bundle, choose **Plugin → Load Plugin Folder...** and select the folder containing `plugin.json`. Do not select an internal `.py` file or import the repository-wide Source code ZIP.

**pyNuD 2.13.0以降**で、必要なZIPを **Plugin → Import Plugin ZIP...** から読み込んでください。専用AIモジュールを含むフォルダー一式がインストールされます。解凍済みの場合は **Plugin → Load Plugin Folder...** で `plugin.json` を含むフォルダーを指定できます。

Use the rebuilt v2.13.0 installer from the current application release. Earlier v2.13.0 installers need to be downloaded and installed again to receive the PyVista plugin-loading fix; the displayed version remains 2.13.0.

本体は公開ページにある差し替え後のv2.13.0を使ってください。先にv2.13.0をダウンロードした場合は、PyVistaプラグイン読み込みの修正を反映するため、再ダウンロード・再インストールが必要です。バージョン表記は2.13.0のままです。

Plugin versions are independent of the host version and this collection tag. Changed public plugins below are **2026.10.4**; unchanged bundles retain **2026.10.1**. Every public plugin is included so the latest-download links remain usable.

## Updated plugins / 更新内容

- **Particle Tracking 2026.10.4**: Codex/Claude numerical AI processing, frame-range Refine, explicit Apply, large image review, small-screen scrolling, trajectory/movie export, and Windows UTF-8/checkpoint and source-byte preservation fixes.
- **Particle Analysis 2026.10.4**: AI review of detections and movie-wide IDs, local refinement and review retention, large image review, corrected rolling-ball subtraction, particle CSV handling/completion notifications, and Windows UTF-8/checkpoint fixes.
- **Dwell Analysis 2026.10.4**: separate movie/event inspection, prompted Refine, manual frame marks, retention of unaffected review decisions, clearer pending/accepted state, shared AI-round limits and validated replay handling.
- **Particle Cluster Analysis 2026.10.4**: corrected ellipse-fit export/import indexing, pixel units for segmentation sigma, accurate tooltips and compact-panel support.

The dedicated AI files are included in the ZIPs. Shared AI connections, numerical runtime and image review are provided by the updated host. AI transmission still requires the user's data-sharing consent; results require scientific review before acceptance.

## Distribution changes / 配布形式

- One complete ZIP per plugin replaces standalone `.py` downloads. Optional resources such as the Simulator Bridge image are included.
- Versions and update dates are recorded in `plugin_index.json` and each bundle's `plugin.json`.
- `SHA256SUMS.txt` covers all plugin ZIPs and the index.
- The current distribution uses bundle folders and ZIPs. Old root-level single-file editions are retained unchanged for existing external links; they are not updated or attached to this release.
- The retired AFM Simulator plugin is replaced by the standalone simulator plus Simulator Bridge.
- Normal Mode Analysis requires pyNuD running in a Python environment with ProDy. The packaged Mac/Windows application explicitly does not enable this plugin.

プラグインは本体インストーラーには含まれません。更新前にプラグインとAIレビュー画面を閉じ、新しいZIPを読み込んでください。本体を更新した場合は再起動してください。

## Verification

ZIP contents are checked byte-for-byte against their declared source files; every bundle is extracted with the actual host importer, and plugin entry points and dedicated helper modules are loaded. The public download assets are checked against SHA-256 digests after upload. These checks validate packaging and imports; they do not claim live-model analysis accuracy or validate every scientific workflow.

Plugin source snapshot: `f162c18`; corrected host installer source: `33c2ed1`.
