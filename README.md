# pyNuD Plugins

Download **one ZIP per plugin** and import it in **pyNuD 2.13.0 or later** using **Plugin → Import Plugin ZIP...**. No manual extraction is needed. Each ZIP contains the complete plugin folder, including its dedicated AI modules and resources.

必要なプラグインのZIPを個別にダウンロードし、**pyNuD 2.13.0以降 → Plugin → Import Plugin ZIP...** で読み込んでください。解凍は不要です。旧 `.py` 単体ファイルでは必要な専用モジュールが不足します。

- For an extracted bundle, use **Load Plugin Folder...** and select the folder containing `plugin.json`. Keep that folder in place while using the plugin; folder loading uses it directly, whereas ZIP import installs a copy in pyNuD’s plugin storage.
- Use the rebuilt v2.13.0 installer from the current application release. If you downloaded v2.13.0 before the plugin-loading fix, download and install it again; the displayed version is unchanged.
- Close the plugin and its AI review windows before importing an updated ZIP. Restart pyNuD after updating the host application.
- **Each plugin has its own version**, shown below and in its window title. The `bundles-2026.10.04` release identifies this collection; it is not a shared plugin version or the application version.
- Do not import GitHub's repository-wide **Source code (zip)** as a plugin.
- Older `.py` downloads remain in [historical releases](https://github.com/uchihast/pyNuD-plugins/releases). Root-level `.py` files are frozen legacy copies retained for existing external links; they do not contain the new bundle updates. Use the ZIP downloads below.

## Downloads / ダウンロード

| Plugin | Version | Updated | Download |
|---|---|---|---|
| AFM Movie Editor | 2026.10.1 | 2026-10-01 | [AFMMovieEditor.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/AFMMovieEditor.zip) |
| Dwell Analysis | 2026.10.4 | 2026-10-04 | [DwellAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/DwellAnalysis.zip) |
| Filament Analysis | 2026.10.1 | 2026-10-01 | [FilamentAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/FilamentAnalysis.zip) |
| Kymograph | 2026.10.1 | 2026-10-01 | [Kymograph.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/Kymograph.zip) |
| L-AFM Analysis | 2026.10.1 | 2026-10-01 | [LAFMAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/LAFMAnalysis.zip) |
| Normal Mode Analysis | 2026.10.1 | 2026-10-01 | [NormalModeAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/NormalModeAnalysis.zip) |
| Particle Analysis | 2026.10.4 | 2026-10-04 | [ParticleAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/ParticleAnalysis.zip) |
| Particle Tracking | 2026.10.4 | 2026-10-04 | [ParticleTracking.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/ParticleTracking.zip) |
| Simulator Bridge | 2026.10.1 | 2026-10-01 | [SimulatorBridge.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/SimulatorBridge.zip) |
| Spot Analysis | 2026.10.1 | 2026-10-01 | [SpotAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/SpotAnalysis.zip) |
| Particle Cluster Analysis | 2026.10.4 | 2026-10-04 | [particle_cluster_analysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/particle_cluster_analysis.zip) |

[Plugin index](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/plugin_index.json) · [SHA-256 checksums](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/SHA256SUMS.txt) · [Bundle guide](PLUGIN_BUNDLES.md) · [Release notes](RELEASE_NOTES.md)

The download links always target the latest collection. Every collection includes unchanged public plugins as well, so all listed links remain available. Future fixes advance only the affected plugin versions.

## Application and help

- [pyNuD application installers](https://github.com/uchihast/pyNuD-installer/releases/latest)
- [D-Lab software page](https://dlab-website-2026.vercel.app/software?section=plugins) (older pages may still describe single-file installation; use the ZIP instructions above).
- The old AFM Simulator plugin has been retired. Use the standalone pyNuD Simulator with **SimulatorBridge.zip** for Live sync.
- **Normal Mode Analysis is not available in packaged Mac/Windows applications.** Run pyNuD in a Python environment with ProDy installed to use this plugin. ZIP import does not install optional dependencies.
