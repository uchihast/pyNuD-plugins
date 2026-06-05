# pyNuD Plugins

pyNuD plugins distributed as standalone `.py` files.

Use pyNuD `Plugin` -> `Load Plugin...` and select the downloaded `.py` file.

## Latest Download Links

These URLs always point to the latest GitHub Release assets.

### Structure / Simulation

- [AFMSimulator.py](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/AFMSimulator.py)  
  Simulates AFM images from PDB/mmCIF files and supports comparison with real AFM images.
- [NormalModeAnalysis.py](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/NormalModeAnalysis.py)  
  Loads PDB files and visualizes molecular flexibility and collective motions using ProDy normal modes.

### Contour / Molecular Analysis

- [FilamentAnalysis.py](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/FilamentAnalysis.py)  
  Extracts filament centerlines and measures contour length using spline interpolation and arc-length integration.

### Time Axis / Kymograph

- [DwellAnalysis.py](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/DwellAnalysis.py)  
  Links frame-by-frame marks, computes dwell times, and exports histogram/CSV results.
- [Kymograph.py](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/Kymograph.py)  
  Creates time-distance kymographs by stacking intensity profiles along line or polyline ROIs.

### L-AFM Analysis

- [LAFMAnalysis.py](https://github.com/uchihast/pyNuD-plugins/releases/latest/download/LAFMAnalysis.py)  
  Localizes intensity peaks in time-series AFM data and reconstructs super-resolution images.

## Raw Main Links

For development snapshots, use:

```text
https://raw.githubusercontent.com/uchihast/pyNuD-plugins/main/<PluginName>.py
```

Release links are recommended for websites because they stay stable across source branch changes.
