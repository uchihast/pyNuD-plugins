"""Numerical starting point for data-dependent dwell analysis.

This file is also supplied to the sandboxed agent. It contains no GUI or file
access. All coordinates refer to the unflipped numerical input (x, y); frame
numbers in results are one-based. Extra filtering is for detection, not display.
"""
import warnings
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage.feature import peak_local_max


# Numerical protocol used for the original reviewed trial. Field boundaries are
# deliberately NOT part of the protocol: they must be measured on each movie.
# Optional reference settings in nm/pixels, not universal molecular thresholds.
REFERENCE_METHOD = dict(plane=True, block_frames=25, high_nm=.9, low_nm=.55,
    radius_px=3., separation_px=6, baseline_percentile=25., registration_min=.7,
    texture_cap_nm=2.7, density_threshold_nm=.8, density_floor=.004,
    bridge_nm=.35, clear_peak_nm=1.3, shape_floor_nm=.5)


def register(reference, image, margin):
    margin = max(1, min(int(margin), (min(image.shape)-3)//2))
    corr = cv2.matchTemplate(image.astype('f4'),
                            reference[margin:-margin, margin:-margin].astype('f4'),
                            cv2.TM_CCOEFF_NORMED)
    if np.std(reference) < 1e-10 or not np.all(np.isfinite(corr)):
        return np.zeros(2), 0.
    y, x = np.unravel_index(corr.argmax(), corr.shape)
    result = []
    for pos, values in ((y, corr[:, x]), (x, corr[y, :])):
        delta = 0.
        if 0 < pos < len(values)-1:
            den = values[pos-1]-2*values[pos]+values[pos+1]
            if den < 0:
                delta = float(np.clip(.5*(values[pos-1]-values[pos+1])/den, -.5, .5))
        result.append(margin-pos-delta)
    return np.array(result), float(corr[y, x])


def level_for_detection(frames, check=lambda: None):
    yy, xx = np.indices(frames.shape[1:])
    design = np.column_stack((xx.ravel()/max(1, xx.max()),
                             yy.ravel()/max(1, yy.max()), np.ones(xx.size)))
    out = []
    for image in frames:
        check()
        values = image.ravel(); mask = np.ones(values.size, bool)
        for _ in range(5):
            fit = np.linalg.lstsq(design[mask], values[mask], rcond=None)[0]
            residual = values-design@fit
            low, high = np.percentile(residual, [10, 65])
            selected = (residual >= low) & (residual <= high)
            if selected.sum() < 4: break
            mask = selected
        corrected = residual.reshape(image.shape)
        out.append(corrected-np.percentile(corrected, 20))
    return np.asarray(out, dtype='f4')


def episodes(signal, valid, high, low, bridge_nm=None):
    present = (signal >= low) & valid
    bridge = np.zeros(len(present), bool)
    bridge_floor = low*.64 if bridge_nm is None else bridge_nm
    bridge[1:-1] = (~present[1:-1] & valid[1:-1] & (signal[1:-1] >= bridge_floor)
                    & (signal[:-2] >= high) & (signal[2:] >= high))
    present |= bridge
    labels, count = ndi.label(present)
    return [(int(ix[0]), int(ix[-1]), int(bridge[ix].sum()))
            for label in range(1, count+1)
            if len(ix := np.flatnonzero(labels == label)) and signal[ix].max() >= high]


def analyze(frames, *, segments=None, plane=True, block_frames=25,
            high_nm=None, low_nm=None, radius_px=3., separation_px=6,
            baseline_percentile=25., registration_min=.7, field_min=.45,
            texture_cap_nm=None, density_threshold_nm=None, density_floor=None,
            bridge_nm=None, clear_peak_nm=None, shape_floor_nm=None, check=lambda: None):
    """Return measurable candidates, never human-approved molecular assignments.

    Agents may change parameters, supply inspected stable ranges (inclusive,
    one-based), or replace the algorithm entirely. No sample-specific boundaries.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) < 2 or min(frames.shape[1:]) < 12:
        raise ValueError('Dwell analysis needs at least two images, at least 12 pixels per axis.')
    n, h, w = frames.shape
    check()
    level = level_for_detection(frames, check) if plane else frames.astype('f4').copy()
    cap = float(texture_cap_nm if texture_cap_nm is not None else np.percentile(level, 65))
    def texture(a):
        check()
        a = np.minimum(ndi.gaussian_filter(a, .8), cap)
        return a-ndi.gaussian_filter(a, 5)
    textures = np.array([texture(a) for a in level])
    if segments is None:
        scores = []
        for i in range(1, n):
            check()
            scores.append(register(textures[i-1], textures[i], max(3, min(h, w)//10))[1])
        cuts = [0] + [i+1 for i, score in enumerate(scores) if score < field_min] + [n]
        segments = [[a+1, b] for a, b in zip(cuts[:-1], cuts[1:])]
    coverage = np.zeros(n, bool); sites = []; events = []; segment_records = []
    for seg_id, (first, last) in enumerate(segments, 1):
        check()
        first, last = int(first), int(last)
        if not 1 <= first <= last <= n or coverage[first-1:last].any():
            raise ValueError('Invalid or overlapping segment ranges.')
        if last-first < 2:
            continue  # insufficient independent observations; explicitly unassessed
        coverage[first-1:last] = True
        image = level[first-1:last]; m = len(image)
        starts = list(range(0, m, max(1, int(block_frames))))
        refs = np.array([texture(np.median(image[k:k+block_frames], axis=0)) for k in starts])
        edges = []
        for j in range(1, len(refs)):
            for gap in (1, 2, 4):
                if j < gap: continue
                delta, score = register(refs[j-gap], refs[j], min(8, 2+gap))
                edges.append((j-gap, j, delta, score))
        trajectory = np.zeros((len(refs), 2))
        if edges:
            design = np.zeros((len(edges), len(refs)-1))
            for k, (i, j, _, _) in enumerate(edges):
                design[k, j-1] = 1
                if i: design[k, i-1] = -1
            displacement = np.array([edge[2] for edge in edges])
            weights = np.array([max(.1, edge[3]) for edge in edges])
            for axis in (0, 1):
                guess = np.linalg.lstsq(design*weights[:, None], displacement[:, axis]*weights, rcond=None)[0]
                fit = least_squares(lambda z: (design@z-displacement[:, axis])*weights,
                                    guess, loss='soft_l1', f_scale=.25)
                trajectory[1:, axis] = fit.x
        local = [register(refs[min(len(refs)-1, i//block_frames)], texture(a), 3)
                 for i, a in enumerate(image)]
        quality = np.array([item[1] for item in local])
        shifts = np.array([trajectory[min(len(refs)-1, i//block_frames)]+item[0]
                           for i, item in enumerate(local)])
        aligned = np.array([ndi.shift(a, shift, order=1, mode='constant', cval=np.nan,
                                      prefilter=False) for a, shift in zip(image, shifts)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            baseline = np.nanpercentile(aligned, baseline_percentile, axis=0)
            residual = aligned-baseline
            residual -= np.nanmedian(residual, axis=2)[:, :, None]
        residual = np.nan_to_num(residual)
        response = np.array([ndi.gaussian_filter(a, 1)-ndi.gaussian_filter(a, 6) for a in residual])
        diff = np.diff(response, axis=0)
        noise = 1.4826*np.median(np.abs(diff-np.median(diff, axis=0)), axis=0)/np.sqrt(2)
        # Noise-derived starting values; AI must inspect event and off-state evidence.
        high = float(high_nm if high_nm is not None else max(.05, 6*np.median(noise)))
        low = float(low_nm if low_nm is not None else .61*high)
        if not 0 < low < high: raise ValueError('Require 0 < low threshold < high threshold.')
        density_cut = high*.89 if density_threshold_nm is None else density_threshold_nm
        density = ndi.gaussian_filter(np.mean(np.maximum(response-density_cut, 0), axis=0), 1)
        points = peak_local_max(density, min_distance=max(2, int(separation_px)),
                                threshold_abs=max(.001, high*.0044) if density_floor is None else density_floor,
                                exclude_border=4)
        yy, xx = np.indices((h, w))
        segment_records.append(dict(id=seg_id, first_frame=first, last_frame=last,
                                    registration_median=float(np.median(quality))))
        for y, x in sorted(points, key=lambda point: (-point[0], point[1])):
            check()
            disk = (yy-y)**2+(xx-x)**2 <= radius_px**2
            signal = np.percentile(response[:, disk], 80, axis=1)
            positions = np.array([x, y])[None, :]-shifts[:, ::-1]
            margin = radius_px+2
            valid = ((positions[:, 0] >= margin) & (positions[:, 0] < w-margin)
                     & (positions[:, 1] >= margin) & (positions[:, 1] < h-margin)
                     & (quality >= registration_min))
            detected = episodes(signal, valid, high, low, bridge_nm)
            if not detected: continue
            ident = f'S{len(sites)+1:03d}'
            site = dict(id=ident, first_frame=first, last_frame=last, radius_px=float(radius_px),
                        display_radius_px=float(radius_px*1.5),
                        positions_px=positions.tolist(), response_nm=signal.tolist(), valid=valid.tolist(),
                        low_nm=low, high_nm=high, noise_nm=float(np.median(noise[disk])))
            sites.append(site)
            for start, end, bridged in detected:
                left = start == 0 or not valid[start-1]
                right = end == m-1 or not valid[end+1]
                peak = start+int(np.argmax(signal[start:end+1])); flags = []
                if end-start+1 < 3: flags.append('short_event')
                if signal[peak] < (high*1.44 if clear_peak_nm is None else clear_peak_nm): flags.append('weak_height_change')
                if bridged: flags.append('one_frame_gap_bridged')
                for side, off, on in [('onset', signal[max(0, start-3):start], signal[start:min(end+1, start+3)]),
                                      ('offset', signal[end+1:end+4], signal[max(start, end-2):end+1])]:
                    if len(off) < 3 or off.max() >= high or np.median(off) >= low:
                        flags.append('no_stable_low_state_at_'+side)
                    censored = left if side == 'onset' else right
                    if not censored and len(off) and np.median(on)-np.median(off) < high*.5:
                        flags.append('gradual_or_uncertain_'+side)
                patch = response[peak, max(0, y-6):y+7, max(0, x-12):x+13]
                labels, _ = ndi.label(patch >= max(high*.55 if shape_floor_nm is None else shape_floor_nm,
                                                  response[peak, y, x]*.4))
                label = labels[y-max(0, y-6), x-max(0, x-12)]
                coords = np.argwhere(labels == label) if label else np.empty((0, 2))
                if len(coords) < 6 or (np.ptp(coords, axis=0)[1]+1)/(np.ptp(coords, axis=0)[0]+1) > 4:
                    flags.append('scan_streak_or_small_feature')
                if left: flags.append('left_censored')
                if right: flags.append('right_censored')
                events.append(dict(site=ident, first_frame=first+start, last_frame=first+end,
                                   peak_frame=first+peak, left_censored=bool(left), right_censored=bool(right),
                                   status='review_required' if flags else 'clear_candidate', flags=flags))
    events.sort(key=lambda event: (event['first_frame'], event['site']))
    for i, event in enumerate(events, 1): event['id'] = f'E{i:04d}'
    return dict(algorithm='Registered temporal baseline and local positive height episodes',
                signal_definition='80th percentile of local height change after temporal baseline, row offset and broad background removal (nm). Cross: measured site; circle: review ROI (1.5 times the measurement radius).',
                parameters=dict(plane_for_detection=plane, block_frames=block_frames, high_nm=high_nm,
                                low_nm=low_nm, radius_px=radius_px, separation_px=separation_px, baseline_percentile=baseline_percentile,
                                registration_min=registration_min, field_min=field_min, texture_cap_nm=cap, segments=[list(v) for v in segments],
                                density_threshold_nm=density_threshold_nm, density_floor=density_floor,
                                bridge_nm=bridge_nm, clear_peak_nm=clear_peak_nm, shape_floor_nm=shape_floor_nm),
                warnings=['Height changes are candidate binding/unbinding, not molecular identity.',
                          'Continuously occupied sites and sub-frame events may be missed. Review all candidates.'],
                assessed_frames=(np.flatnonzero(coverage)+1).tolist(), segments=segment_records,
                sites=sites, events=events)
