# pyNuD plugin bundles

Download the ZIP for one plugin and choose **Plugin → Import Plugin ZIP...**.
pyNuD validates and installs the complete folder; manual extraction is not required.
For an extracted folder, choose **Plugin → Load Plugin Folder...**.
Use pyNuD 2.13.0 or later (plugin bundle API 1). Update older hosts and restart first.
Interactive Drift Compensation, Reference Object Cropper and L-AFM Analysis 2026.10.20 require pyNuD 2.13.1 or later.
Each plugin has its own version in plugin.json; the collection tag is not its version.
Plugin versions use the revision date in Japan (JST): YYYY.M.D, without zero padding.
Additional revisions on the same date append .1, .2, etc.; a new date resets this suffix.
The updated field uses the same date as YYYY-MM-DD. Unchanged plugins keep their versions.
Close the plugin and its AI review windows before importing an updated ZIP.
Each ZIP import gets a separate folder in the user application data area; existing files are preserved.
Reopening reloads its dedicated AI code; shared host updates require restarting pyNuD.
All plugins, including those with one implementation file, use complete folders.

`_legacy_single_files/` preserves old export files for maintainer review; do not install those copies.

| Plugin | Version | Updated | Download |
|---|---|---|---|
| AFM Movie Editor | 2026.10.1 | 2026-10-01 | [AFMMovieEditor.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/AFMMovieEditor.zip) |
| Dwell Analysis | 2026.10.4 | 2026-10-04 | [DwellAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/DwellAnalysis.zip) |
| Filament Analysis | 2026.10.1 | 2026-10-01 | [FilamentAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/FilamentAnalysis.zip) |
| Interactive Drift Compensation | 2026.10.20 | 2026-10-20 | [InteractiveDriftCompensation.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/InteractiveDriftCompensation.zip) |
| Kymograph | 2026.10.1 | 2026-10-01 | [Kymograph.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/Kymograph.zip) |
| L-AFM Analysis | 2026.10.20 | 2026-10-20 | [LAFMAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/LAFMAnalysis.zip) |
| Normal Mode Analysis | 2026.10.1 | 2026-10-01 | [NormalModeAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/NormalModeAnalysis.zip) |
| Particle Analysis | 2026.10.4 | 2026-10-04 | [ParticleAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/ParticleAnalysis.zip) |
| Particle Tracking | 2026.10.4 | 2026-10-04 | [ParticleTracking.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/ParticleTracking.zip) |
| Reference Object Cropper | 2026.10.20 | 2026-10-20 | [ReferenceObjectCropper.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/ReferenceObjectCropper.zip) |
| Simulator Bridge | 2026.10.1 | 2026-10-01 | [SimulatorBridge.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/SimulatorBridge.zip) |
| Spot Analysis | 2026.10.1 | 2026-10-01 | [SpotAnalysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/SpotAnalysis.zip) |
| Particle Cluster Analysis | 2026.10.4 | 2026-10-04 | [particle_cluster_analysis.zip](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/particle_cluster_analysis.zip) |
