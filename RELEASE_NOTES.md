# Plugin ZIP bundles — 2026-10-04

## Installation / インストール

Use **pyNuD 2.13.0 or later**. Download one plugin ZIP below and choose **Plugin → Import Plugin ZIP...**. Do not extract it manually, select an internal `.py` file, or import the repository-wide Source code ZIP.

**pyNuD 2.13.0以降**で、必要なZIPを **Plugin → Import Plugin ZIP...** から読み込んでください。専用AIモジュールを含むフォルダー一式がインストールされます。本体v2.13.0には、PyVistaを使うプラグインの読み込みで再帰エラーになる問題の修正が含まれます。

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

プラグインは本体インストーラーには含まれません。更新前にプラグインとAIレビュー画面を閉じ、新しいZIPを読み込んでください。本体を更新した場合は再起動してください。

## Verification

ZIP contents are checked byte-for-byte against their declared source files; every bundle is extracted with the actual host importer, and plugin entry points and dedicated helper modules are loaded. The public download assets are checked against SHA-256 digests after upload. These checks validate packaging and imports; they do not claim live-model analysis accuracy or validate every scientific workflow.

Plugin source snapshot: `f162c18`; corrected host installer source: `33c2ed1`.
