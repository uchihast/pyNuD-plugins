#!/usr/bin/env python3
"""File-based bridge from pyNuD Simulator to an optional AFMfit install.

AFMfit itself is not bundled with pyNuD Simulator.  This script is executed by
the Python interpreter selected by the user and imports AFMfit there.  The two
programs exchange only an input PDB, an NPZ height map, an output PDB, and JSON
metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def emit_progress(stage: str, percent: float, message: str) -> None:
    print(
        "PYNUD_PROGRESS "
        + json.dumps(
            {
                "stage": str(stage),
                "percent": float(percent),
                "message": str(message),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def probe_environment() -> dict:
    import afmfit
    from afmfit.fitting import Fitter  # noqa: F401
    from afmfit.nma import NormalModesRTB  # noqa: F401
    from afmfit.pdbio import PDB  # noqa: F401
    from afmfit.simulator import AFMSimulator  # noqa: F401
    from afmfit.utils import get_nolb_path

    nolb_path = Path(get_nolb_path()).resolve()
    return {
        "afmfit_version": str(getattr(afmfit, "__version__", "unknown")),
        "afmfit_path": str(Path(afmfit.__path__[0]).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "nolb_path": str(nolb_path),
        "nolb_exists": bool(nolb_path.is_file()),
        "nolb_executable": bool(os.access(nolb_path, os.X_OK)),
        "license": "GPL-3.0-or-later",
    }


def prepare_image(input_path: Path) -> tuple[np.ndarray, float, dict]:
    """Convert pyNuD's nm image into AFMfit's square Å image convention."""
    with np.load(input_path, allow_pickle=False) as payload:
        height_nm = np.asarray(payload["height_nm"], dtype=np.float64)
        pixel_x_nm = float(payload["pixel_x_nm"].item())
        pixel_y_nm = float(payload["pixel_y_nm"].item())
    if height_nm.ndim != 2 or height_nm.size < 16:
        raise RuntimeError("AFMfit input image must be a non-empty 2D array.")
    if pixel_x_nm <= 0.0 or pixel_y_nm <= 0.0:
        raise RuntimeError("AFMfit input pixel sizes must be positive.")

    valid = np.isfinite(height_nm) & (height_nm > -1e8)
    values = height_nm[valid]
    if values.size < 16:
        raise RuntimeError("AFMfit input image has too few valid pixels.")
    baseline_nm = float(np.percentile(values, 5.0))
    image_nm = np.zeros(height_nm.shape, dtype=np.float64)
    image_nm[valid] = np.maximum(height_nm[valid] - baseline_nm, 0.0)

    # AFMfit supports one isotropic pixel size.  Resample only when the source
    # X/Y pixel sizes differ materially.
    isotropic_nm = 0.5 * (pixel_x_nm + pixel_y_nm)
    if (
        abs(pixel_x_nm - isotropic_nm) / isotropic_nm > 1e-3
        or abs(pixel_y_nm - isotropic_nm) / isotropic_nm > 1e-3
    ):
        from scipy.ndimage import zoom

        image_nm = zoom(
            image_nm,
            (pixel_y_nm / isotropic_nm, pixel_x_nm / isotropic_nm),
            order=1,
            mode="constant",
            cval=0.0,
        )

    rows, columns = image_nm.shape
    size = max(32, rows, columns)
    size = int(np.ceil(size / 8.0) * 8)
    padded_nm = np.zeros((size, size), dtype=np.float32)
    row_start = (size - rows) // 2
    column_start = (size - columns) // 2
    padded_nm[
        row_start:row_start + rows,
        column_start:column_start + columns,
    ] = image_nm

    # Equivalent to AFMfit ImageSet.arr2img(..., unit="nm") for one image:
    # convert heights to Å, transpose X/Y, and reverse the second image axis.
    afm_image_angstrom = (
        (padded_nm * 10.0).T[:, ::-1]
    ).astype(np.float32, copy=False)
    metadata = {
        "source_shape": [int(height_nm.shape[0]), int(height_nm.shape[1])],
        "fit_shape": [int(size), int(size)],
        "pixel_size_angstrom": float(isotropic_nm * 10.0),
        "baseline_nm": baseline_nm,
    }
    return afm_image_angstrom, float(isotropic_nm * 10.0), metadata


def run_fit(args: argparse.Namespace) -> dict:
    from afmfit.fitting import Fitter
    from afmfit.nma import NormalModesRTB
    from afmfit.pdbio import PDB
    from afmfit.simulator import AFMSimulator

    started_at = time.monotonic()
    emit_progress("input", 3.0, "Reading PDB and AFM height map")
    input_pdb = Path(args.input_pdb).resolve()
    output_pdb = Path(args.output_pdb).resolve()
    result_json = Path(args.result_json).resolve()
    image, pixel_size_angstrom, image_metadata = prepare_image(
        Path(args.input_image).resolve()
    )
    pdb = PDB(str(input_pdb))
    if pdb.n_atoms < 4:
        raise RuntimeError("AFMfit requires at least four PDB atoms.")
    input_center_angstrom = np.mean(
        np.asarray(pdb.coords, dtype=np.float64),
        axis=0,
    )
    # AFMfit's simulator is centered on the origin, and its own tutorial/tests
    # center the PDB before projection matching.  pyNuD restores the original
    # coordinate frame after fitting with Kabsch alignment.
    pdb.center()

    emit_progress(
        "nma",
        12.0,
        f"Calculating {int(args.nmodes)} nonlinear RTB modes with NOLB",
    )
    nma = NormalModesRTB.calculate_NMA(
        pdb=pdb,
        nmodes=int(args.nmodes),
        cutoff=float(args.cutoff_angstrom),
    )
    simulator = AFMSimulator(
        size=int(image.shape[0]),
        vsize=float(pixel_size_angstrom),
        beta=1.0,
        sigma=float(args.sigma_angstrom),
    )

    emit_progress(
        "rigid",
        30.0,
        "Refining the Estimate Pose orientation with AFMfit projection matching",
    )
    fitter = Fitter(pdb=pdb, imgs=np.asarray([image]), simulator=simulator)
    half_z = 0.5 * float(args.z_shift_range_angstrom)
    z_shift_range = np.linspace(
        -half_z,
        half_z,
        max(2, int(args.z_shift_points)),
    )
    angular_distance = max(2, int(round(float(args.angular_distance_deg))))
    fitter.fit_rigid(
        n_cpu=max(1, int(args.n_cpu)),
        angular_dist=angular_distance,
        verbose=True,
        zshift_range=z_shift_range,
        init_zshift=None,
        near_angle=[0.0, 0.0, 0.0],
        near_angle_cutoff=max(
            float(args.rigid_angle_limit_deg),
            float(angular_distance),
        ),
        select_view_group=True,
        true_zshift=False,
        metric="rmsdp",
    )

    emit_progress(
        "flexible",
        62.0,
        f"Running {int(args.iterations)} AFMfit flexible-fitting iterations",
    )
    fitter.fit_flexible(
        n_cpu=max(1, int(args.n_cpu)),
        nma=nma,
        verbose=True,
        n_best_views=max(1, int(args.n_best_views)),
        dist_views=max(1.0, float(args.view_separation_deg)),
        n_iter=max(1, int(args.iterations)),
        lambda_r=float(args.regularization_lambda),
        lambda_f=float(args.regularization_lambda),
    )

    emit_progress("output", 94.0, "Writing fitted PDB and result metadata")
    fitted_coords = np.asarray(fitter.flexible_coords[0], dtype=np.float64)
    if fitted_coords.shape != pdb.coords.shape or not np.all(
        np.isfinite(fitted_coords)
    ):
        raise RuntimeError("AFMfit returned invalid fitted coordinates.")
    fitted_pdb = pdb.copy()
    fitted_pdb.coords = fitted_coords.astype(np.float32)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    fitted_pdb.write_pdb(str(output_pdb))

    best_flexible_index = int(np.argmin(fitter.flexible_scores[0]))
    result = {
        "afmfit_version": str(probe_environment()["afmfit_version"]),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "atom_count": int(pdb.n_atoms),
        "input_center_angstrom": input_center_angstrom.tolist(),
        "image": image_metadata,
        "rigid_score": float(np.min(fitter.rigid_scores[0])),
        "flexible_score": float(
            fitter.flexible_scores[0, best_flexible_index]
        ),
        "afmfit_structural_rmsd_angstrom": float(
            fitter.flexible_rmsds[0, best_flexible_index]
        ),
        "rigid_angle_deg": np.asarray(
            fitter.rigid_angles[0, 0],
            dtype=float,
        ).tolist(),
        "rigid_shift_angstrom": np.asarray(
            fitter.rigid_shifts[0, 0],
            dtype=float,
        ).tolist(),
        "flexible_angle_deg": np.asarray(
            fitter.flexible_angles[0],
            dtype=float,
        ).tolist(),
        "flexible_shift_angstrom": np.asarray(
            fitter.flexible_shifts[0],
            dtype=float,
        ).tolist(),
        "settings": {
            "n_cpu": int(args.n_cpu),
            "nmodes": int(args.nmodes),
            "cutoff_angstrom": float(args.cutoff_angstrom),
            "sigma_angstrom": float(args.sigma_angstrom),
            "angular_distance_deg": float(angular_distance),
            "rigid_angle_limit_deg": float(args.rigid_angle_limit_deg),
            "z_shift_range_angstrom": float(args.z_shift_range_angstrom),
            "z_shift_points": int(args.z_shift_points),
            "n_best_views": int(args.n_best_views),
            "view_separation_deg": float(args.view_separation_deg),
            "iterations": int(args.iterations),
            "regularization_lambda": float(args.regularization_lambda),
        },
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    emit_progress("complete", 100.0, "AFMfit completed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="External AFMfit bridge for pyNuD Simulator"
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input-pdb")
    parser.add_argument("--input-image")
    parser.add_argument("--output-pdb")
    parser.add_argument("--result-json")
    parser.add_argument("--n-cpu", type=int, default=2)
    parser.add_argument("--nmodes", type=int, default=10)
    parser.add_argument("--cutoff-angstrom", type=float, default=8.0)
    parser.add_argument("--sigma-angstrom", type=float, default=4.0)
    parser.add_argument("--angular-distance-deg", type=float, default=10.0)
    parser.add_argument("--rigid-angle-limit-deg", type=float, default=25.0)
    parser.add_argument("--z-shift-range-angstrom", type=float, default=20.0)
    parser.add_argument("--z-shift-points", type=int, default=5)
    parser.add_argument("--n-best-views", type=int, default=5)
    parser.add_argument("--view-separation-deg", type=float, default=15.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--regularization-lambda", type=float, default=25.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.probe:
            print(json.dumps(probe_environment(), sort_keys=True), flush=True)
            return 0
        required = (
            "input_pdb",
            "input_image",
            "output_pdb",
            "result_json",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise RuntimeError(
                "Missing required AFMfit bridge arguments: "
                + ", ".join(missing)
            )
        run_fit(args)
        return 0
    except Exception as exc:
        emit_progress("error", 100.0, str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
