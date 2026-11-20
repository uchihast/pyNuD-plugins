"""Irregular molecule detection core (no Qt). / 不定形分子の検出の計算部（Qt 非依存）

Method "E" chosen by visual comparison on 2026-11-19 (HS-AFM, mRNA clumps with single strands).
2026-11-19 に目視比較で採用した方法 E。

Steps / 手順
  1. Background: masked quadratic surface + per-scan-line median (iterated), or a grey-opening
     top-hat for raw wide-field images. Drift-correction padding is excluded.
     背景: 分子を除いた 2 次曲面 + 走査ラインの中央値（反復）。未処理の広視野像はトップハット。
  2. Evidence map z = max(S/N of smoothed height, S/N of Sato ridge response), both normalised by
     the robust sigma of background pixels.
     証拠マップ: 平滑高さとリッジ応答の S/N の大きい方。背景のロバスト σ で規格化。
  3. Hysteresis threshold (k_low, k_high); keep components containing a strong height core.
     2 段閾値。強い芯を含む成分だけ残す。
  4. Group components closer than group_gap_nm into one molecule.
     group_gap_nm 以内の成分を同じ分子にまとめる。
  5. Redraw the outline on 3x3-median height: drop pixels below the noise level and valley
     bottoms between strands, keep strand centre lines.
     3x3 メディアン高さで輪郭を引き直し、鎖の間の谷を除く。鎖の中心線は残す。
  6. Per molecule keep the component holding the bright core, plus islands within
     island_gap_nm whose skeleton is at least min_island_length_nm.
     明るい本体と直結する成分 + 近くの長い島だけ残す。
  7. Optionally drop molecules touching the image (or padding) border.
     画像の端に接する分子を除外（任意）。
Volume = sum of height inside the mask x pixel area. / 体積 = マスク内の高さの和 × 画素面積。

Generality / 汎用性
  Every length is given in nm and converted with the pixel size, so the same settings work for
  any pixel size and field of view. Images with non-square pixels are resampled to square pixels
  (the smaller of the two sizes) before detection; the result carries the pixel size it uses.
  長さはすべて nm で指定し、画素サイズで換算する。正方でない画素の像は、小さい方の画素サイズの
  正方画素に補間してから検出する。結果には使った画素サイズを持たせる。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


@dataclass
class DetectionParams:
    background: str = "polynomial"          # "polynomial" | "tophat"
    tophat_radius_nm: float = 8.0
    strand_width_min_nm: float = 1.0
    strand_width_max_nm: float = 2.0
    k_low: float = 2.0
    k_high: float = 4.0
    core_k: float = 6.0                     # a molecule needs smoothed height above this (σ); 0 = no core
    min_area_nm2: float = 30.0
    largest_only: bool = False
    group_gap_nm: float = 4.0
    cut_gaps: bool = True
    island_gap_nm: float = 3.0
    min_island_length_nm: float = 12.0
    exclude_edge: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DetectionResult:
    height: np.ndarray                      # background-subtracted height used for measurement (nm)
    labels: np.ndarray                      # 0 = background, 1..N = molecules
    valid: np.ndarray                       # False on drift-correction padding
    sigma_nm: float
    molecules: List[Dict] = field(default_factory=list)
    excluded_edge: int = 0
    px_nm: float = 1.0                      # pixel size of height / labels (after any resampling)
    py_nm: float = 1.0


EDGE_BLIND_PX = 3   # minimum width (px) where filters are unreliable near the image / padding border
SQUARE_TOLERANCE = 0.01  # pixels whose x and y sizes differ by more than this fraction are resampled


# ----------------------------------------------------------------------------- helpers
def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 1.0
    s = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return s if s > 0 else (float(np.std(x)) or 1.0)


def _px(nm: float, px_nm: float, minimum: int = 1) -> int:
    return max(minimum, int(round(nm / px_nm)))


def _odd(nm: float, px_nm: float) -> int:
    """Odd filter size in pixels for a length in nm (1 means no filtering)."""
    return int(round(nm / px_nm)) | 1


def _median(h: np.ndarray, px_nm: float) -> np.ndarray:
    """About 3 nm median. Skipped when that is a single pixel, so 1-px strands survive."""
    size = _odd(3.0, px_nm)
    return ndi.median_filter(h, size) if size > 1 else h.copy()


def valid_mask(a: np.ndarray, min_plateau: int = None) -> np.ndarray:
    """Drift-correction padding = constant-value plateaus touching the border.

    A plateau must be at least as large as the shorter image side (default), so a padding strip
    one pixel wide is found in images of any size, while short runs of equal values are not.
    """
    if min_plateau is None:
        min_plateau = max(20, min(a.shape))
    values, counts = np.unique(a, return_counts=True)
    bad = np.zeros(a.shape, bool)
    for val in values[counts > min_plateau]:
        lab = measure.label(a == val, connectivity=1)
        border = set(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]].tolist()) - {0}
        for l in border:
            region = lab == l
            if region.sum() > min_plateau:
                bad |= region
    return ~ndi.binary_dilation(bad, iterations=2)


def _line_median(b: np.ndarray, bg: np.ndarray) -> np.ndarray:
    b = b.copy()
    for i in range(b.shape[0]):
        r = b[i][bg[i]]
        if r.size > 10:
            b[i] -= np.median(r)
    return b


def level_polynomial(a: np.ndarray, exclude: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Quadratic surface + per-line median, fitted on valid & ~exclude pixels."""
    ny, nx = a.shape
    # coordinates scaled to [-1, 1] keep the fit well conditioned for large images
    yy, xx = np.mgrid[-1:1:ny * 1j, -1:1:nx * 1j]
    bg = valid & ~exclude
    if bg.sum() < 20:
        bg = valid
    A = np.c_[np.ones(bg.sum()), xx[bg], yy[bg], xx[bg] ** 2, yy[bg] ** 2, xx[bg] * yy[bg]]
    c, *_ = np.linalg.lstsq(A, a[bg], rcond=None)
    b = a - (c[0] + c[1] * xx + c[2] * yy + c[3] * xx ** 2 + c[4] * yy ** 2 + c[5] * xx * yy)
    b = _line_median(b, bg)
    b[~valid] = 0.0
    return b


def tophat_background(a: np.ndarray, valid: np.ndarray, radius_px: int, px_nm: float) -> np.ndarray:
    """Remove structures wider than ~2*radius with a grey opening of a median-filtered image."""
    b = a.astype(float).copy()
    b[~valid] = np.median(a[valid])
    b = _line_median(b, valid)
    bg = ndi.grey_opening(_median(b, px_nm), footprint=morphology.disk(radius_px))
    bg = ndi.gaussian_filter(bg, radius_px / 2.0)
    t = b - bg
    t[~valid] = 0.0
    return t


def _evidence(h, bg, widths_px, smooth_px):
    hs = ndi.gaussian_filter(h, smooth_px)
    R = filters.sato(h, sigmas=widths_px, black_ridges=False)
    zh = (hs - np.median(hs[bg])) / robust_sigma(hs[bg])
    zr = (R - np.median(R[bg])) / robust_sigma(R[bg])
    return np.maximum(zh, zr), zh


def _remove_small(mask, max_size):
    if max_size < 1:
        return mask
    lab = measure.label(mask, connectivity=2)
    sizes = np.bincount(lab.ravel())
    small = sizes <= max_size
    small[0] = False
    return mask & ~small[lab]


def edge_blind_px(p: DetectionParams, px_nm: float) -> int:
    """Border width (px) where the widest ridge filter still sees the image edge."""
    return max(EDGE_BLIND_PX, int(np.ceil(1.5 * max(p.strand_width_min_nm, p.strand_width_max_nm) / px_nm)))


def detect_candidates(a: np.ndarray, px_nm: float, p: DetectionParams, V: np.ndarray, n_iter: int = 4):
    """Steps 1-3. Returns (height, mask, inner_valid)."""
    Ve = ndi.binary_erosion(V, iterations=edge_blind_px(p, px_nm))
    if Ve.sum() < 50:
        Ve = V
    w = np.linspace(p.strand_width_min_nm, max(p.strand_width_min_nm, p.strand_width_max_nm), 3)
    widths = tuple(sorted({max(0.7, x / px_nm) for x in w}))
    smooth = max(0.7, 1.0 / px_nm)
    min_px = int(p.min_area_nm2 / px_nm ** 2)
    mask = np.zeros(a.shape, bool)
    h = a.astype(float)
    tophat = p.background == "tophat"
    if tophat:
        h = tophat_background(a, V, _px(p.tophat_radius_nm, px_nm, 2), px_nm)
    for it in range(n_iter):
        excl = ndi.binary_dilation(mask, iterations=_px(2.0, px_nm)) if mask.any() else mask
        if tophat:
            hh = _line_median(h, Ve & ~excl)
        else:
            hh = level_polynomial(a, excl, V)
            if it == 0:  # bootstrap: exclude obvious bright structures before the first statistics
                b0 = Ve & ~excl
                excl = excl | (hh > np.median(hh[b0]) + 2.5 * robust_sigma(hh[b0]))
                hh = level_polynomial(a, excl, V)
        bg = Ve & ~excl
        if bg.sum() < 20:
            bg = Ve
        z, zh = _evidence(hh, bg, widths, smooth)
        z[~Ve] = 0
        cand = filters.apply_hysteresis_threshold(z, p.k_low, p.k_high)
        cand = _remove_small(cand, min_px)
        lab = measure.label(cand, connectivity=2)
        if p.core_k > 0:
            seeds = (zh > p.core_k) & (lab > 0)
            mask = np.isin(lab, np.unique(lab[seeds])) & (lab > 0)
        else:
            mask = lab > 0
    return hh, mask, Ve


def refine_boundary(h, mask, V, px_nm=1.0, k_edge=2.0, k_ridge=3.0, valley_frac=0.45):
    """Step 5: outline on median-filtered height, gaps between strands removed, strand centre lines kept."""
    if not mask.any():
        return mask
    hm = _median(h, px_nm)
    bgm = V & ~ndi.binary_dilation(mask, iterations=_px(3.0, px_nm))
    if bgm.sum() < 20:
        bgm = V & ~mask
    t_edge = np.median(hm[bgm]) + k_edge * robust_sigma(hm[bgm])
    region = ndi.binary_dilation(mask, iterations=1) & V
    keep = region & (hm > t_edge)
    dark = filters.sato(hm, sigmas=(max(0.7, 1.0 / px_nm),), black_ridges=True)
    zd = (dark - np.median(dark[bgm])) / robust_sigma(dark[bgm])
    loc = ndi.maximum_filter(hm, size=max(3, _px(5, px_nm) | 1))
    keep &= ~((zd > 3) & (hm < valley_frac * loc))
    R = filters.sato(h, sigmas=tuple(sorted({max(0.7, 1.0 / px_nm), max(0.7, 1.5 / px_nm)})), black_ridges=False)
    zr = (R - np.median(R[bgm])) / robust_sigma(R[bgm])
    ridge = region & (zr > k_ridge) & (h > np.median(h[bgm]))
    core = ndi.binary_dilation(morphology.skeletonize(ridge), iterations=1) & ridge
    keep |= core
    return _remove_small(keep, int(8 / px_nm ** 2) - 1)


# Distances between pieces are always measured the same way: the Euclidean distance between
# the nearest pixel centres of the two pieces, in pixels. 断片間の距離は、どこでも最も近い画素中心どうし
# のユークリッド距離で測る（以前はモードごとに縦横だけの数え方が混ざっていた）。
def _within(mask: np.ndarray, dist_px: float) -> np.ndarray:
    """Pixels whose centre lies at most dist_px from a pixel centre of mask."""
    if not mask.any():
        return mask.copy()
    return ndi.distance_transform_edt(~mask) <= dist_px + 1e-9


def group_pieces(mask: np.ndarray, gap_px: float) -> np.ndarray:
    """Label molecules: pieces at most gap_px apart (directly or through others) share a label."""
    lab = measure.label(mask, connectivity=2)
    n = int(lab.max())
    if n <= 1 or gap_px <= 0:
        return lab
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    owner = lab[iy, ix]  # nearest piece of every pixel; neighbouring owners meet halfway between pieces
    parent = list(range(n + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reach = gap_px / 2.0 + 1.0
    for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        cand = (owner[a] != owner[b]) & (dist[a] <= reach) & (dist[b] <= reach)
        if not cand.any():
            continue
        d = np.hypot(iy[a][cand] - iy[b][cand], ix[a][cand] - ix[b][cand])
        for p, q in set(zip(owner[a][cand][d <= gap_px + 1e-9].tolist(), owner[b][cand][d <= gap_px + 1e-9].tolist())):
            parent[find(p)] = find(q)
    roots = np.array([find(i) for i in range(n + 1)])
    _, relabel = np.unique(roots, return_inverse=True)
    return np.where(lab > 0, relabel[lab], 0)


def main_component(mask, gap_px, min_px):
    """Largest component plus components (>= min_px) within gap_px of it."""
    lab = measure.label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    reach = _within(lab == sizes.argmax(), gap_px)
    ids = np.unique(lab[reach & (lab > 0)])
    ids = [l for l in ids if sizes[l] >= min_px]
    return np.isin(lab, ids)


def keep_core_and_long_islands(mask, h, px_nm, gap_nm, min_len_nm):
    """Step 6: component holding the brightest point + long islands within gap_nm."""
    lab = measure.label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    hm = ndi.gaussian_filter(h, max(0.7, 1.0 / px_nm))
    hm = np.where(mask, hm, -np.inf)
    core = lab == lab[np.unravel_index(np.argmax(hm), hm.shape)]
    reach = _within(core, gap_nm / px_nm)
    out = core.copy()
    rest = measure.label(mask & ~core, connectivity=2)
    for l in range(1, rest.max() + 1):
        isl = rest == l
        if (isl & reach).any() and morphology.skeletonize(isl).sum() * px_nm >= min_len_nm:
            out |= isl
    return out


_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def prune_spurs(sk: np.ndarray, length_px: int) -> np.ndarray:
    """Remove side branches shorter than length_px; keep main lines at full length.

    End points are peeled length_px times, then the lines left are grown back along the
    original skeleton by the same number of steps, which restores trimmed main-line ends.
    """
    if length_px < 1 or not sk.any():
        return sk
    cur = sk.copy()
    for _ in range(length_px):
        ends = cur & (ndi.convolve(cur.astype(np.uint8), _NEIGHBOURS, mode="constant") <= 1)
        if not ends.any() or ends.sum() == cur.sum():
            break
        cur &= ~ends
    if not cur.any():
        return sk  # the whole skeleton is shorter than a spur: keep it
    ends = cur & (ndi.convolve(cur.astype(np.uint8), _NEIGHBOURS, mode="constant") <= 1)
    grow = ends
    for _ in range(length_px):
        grow = ndi.binary_dilation(grow, structure=np.ones((3, 3), bool)) & sk & ~cur
        if not grow.any():
            break
        cur |= grow
    return cur


def _fill_small_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    """Fill holes (background not connected to the image border) of fewer than max_px pixels.

    Written out instead of skimage's remove_small_holes, whose arguments differ between versions.
    """
    holes = measure.label(~mask, connectivity=1)
    border = np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    sizes = np.bincount(holes.ravel())
    small = sizes < max_px
    small[0] = False
    small[border] = False
    return mask | small[holes]


def skeleton_length(mask: np.ndarray, px_nm: float, spur_nm: float = 4.0, hole_nm2: float = 16.0) -> float:
    """Length of the pruned skeleton: 1 px per straight step and √2 px per diagonal step.

    Holes smaller than hole_nm2 (noise inside a clump) are filled and side branches shorter than
    spur_nm (a ragged outline) are removed first; otherwise finer pixels give longer skeletons.
    Larger holes, such as the gap between two strands, are kept. A diagonal step is counted only
    where the two pixels have no common straight neighbour, so corners are not counted twice.
    """
    filled = _fill_small_holes(mask, max(1, int(hole_nm2 / px_nm ** 2)))
    sk = prune_spurs(morphology.skeletonize(filled), _px(spur_nm, px_nm))
    if not sk.any():
        return 0.0
    p = np.pad(sk, 1)
    c = p[1:-1, 1:-1]
    right, down = p[1:-1, 2:], p[2:, 1:-1]
    straight = (c & right).sum() + (c & down).sum()
    dr, dl = p[2:, 2:], p[2:, :-2]
    left = p[1:-1, :-2]
    diag = (c & dr & ~right & ~down).sum() + (c & dl & ~left & ~down).sum()
    steps = straight + np.sqrt(2.0) * diag
    return float(max(steps, 1.0) * px_nm)


def _resample_square(a: np.ndarray, V: np.ndarray, px: float, py: float):
    """Resample to square pixels of the smaller size. Returns (a, V, size)."""
    s = min(px, py)
    zoom = (py / s, px / s)
    a2 = ndi.zoom(a, zoom, order=1, mode="nearest")
    V2 = ndi.zoom(V.astype(np.uint8), zoom, order=0, mode="nearest").astype(bool)
    n = (min(a2.shape[0], V2.shape[0]), min(a2.shape[1], V2.shape[1]))
    return a2[:n[0], :n[1]], V2[:n[0], :n[1]], s


def touches_edge(mask, V, margin_px=1):
    edge = np.zeros(mask.shape, bool)
    edge[:margin_px] = edge[-margin_px:] = True
    edge[:, :margin_px] = edge[:, -margin_px:] = True
    edge |= ~ndi.binary_erosion(V, iterations=margin_px)
    return bool((mask & edge).any())


# ----------------------------------------------------------------------------- main entry
def analyze(height: np.ndarray, x_nm_per_px: float, y_nm_per_px: float = None,
            params: DetectionParams = None) -> DetectionResult:
    """Detect molecules in one height frame (nm) and measure them."""
    p = params or DetectionParams()
    a = np.asarray(height, float)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).any():
        raise ValueError("A 2-D height image is required.")
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    px = float(x_nm_per_px)
    py = float(y_nm_per_px or x_nm_per_px)
    if not (px > 0 and py > 0):
        raise ValueError("The pixel size must be positive.")
    V = valid_mask(a)  # before any resampling: padding plateaus are exact constants only here
    if abs(py / px - 1.0) > SQUARE_TOLERANCE:
        a, V, px = _resample_square(a, V, px, py)
        py = px
    pxa = px * py
    h, m, Ve = detect_candidates(a, px, p, V)
    edge_px = edge_blind_px(p, px)

    # Step 4: molecules
    # the same grouping in both modes; "largest" then keeps the molecule holding the largest piece
    glab = group_pieces(m, p.group_gap_nm / px)
    groups = [glab == g for g in range(1, glab.max() + 1)]
    if p.largest_only and groups:
        pieces = measure.label(m, connectivity=2)
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        groups = [glab == glab[pieces == sizes.argmax()][0]]
    union = np.zeros(a.shape, bool)
    for g in groups:
        union |= g

    # final background with the final exclusion (polynomial mode); top-hat image is used as is
    if p.background != "tophat":
        h = level_polynomial(a, ndi.binary_dilation(union, iterations=_px(3.0, px)), V)
    bgfinal = Ve & ~ndi.binary_dilation(union, iterations=_px(3.0, px))
    sigma = robust_sigma(h[bgfinal]) if bgfinal.sum() > 20 else robust_sigma(h[Ve])

    refined = refine_boundary(h, union, V, px) if p.cut_gaps else union
    labels = np.zeros(a.shape, np.int32)
    molecules, excluded = [], 0
    for g in groups:
        rg = refined & ndi.binary_dilation(g, iterations=1)
        if not rg.any():
            continue
        rg = main_component(rg, p.island_gap_nm / px, max(1, int(6 / pxa)))
        rg = keep_core_and_long_islands(rg, h, px, p.island_gap_nm, p.min_island_length_nm)
        if not rg.any() or rg.sum() * pxa < p.min_area_nm2:
            continue
        # detection ignores the outer edge_px pixels, so an object cut by the edge ends there
        if p.exclude_edge and touches_edge(rg, V, margin_px=edge_px + 1):
            excluded += 1
            continue
        labels[rg & (labels == 0)] = len(molecules) + 1
        hm = h[rg]
        ys, xs = np.nonzero(rg)
        molecules.append(dict(
            id=len(molecules) + 1,
            area_nm2=float(rg.sum() * pxa),
            volume_nm3=float(hm.sum() * pxa),
            max_height_nm=float(hm.max()),
            mean_height_nm=float(hm.mean()),
            skeleton_length_nm=skeleton_length(rg, px, spur_nm=max(4.0, 2.0 * p.strand_width_max_nm),
                                               hole_nm2=(2.0 * max(2.0, p.strand_width_max_nm)) ** 2),
            n_regions=int(measure.label(rg, connectivity=2).max()),
            x_nm=float((xs.mean() + 0.5) * px), y_nm=float((ys.mean() + 0.5) * py),  # pixel centres
        ))
    if p.largest_only and len(molecules) > 1:
        molecules = molecules[:1]
        labels[labels > 1] = 0
    return DetectionResult(height=h, labels=labels, valid=V, sigma_nm=float(sigma),
                           molecules=molecules, excluded_edge=excluded, px_nm=px, py_nm=py)


__all__ = ["DetectionParams", "DetectionResult", "analyze"]
