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
