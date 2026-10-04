"""Reusable numerical stages from the interactive dwell trial.

Input is the captured main-window numerical movie. Detection-only leveling does
not alter review images. Prepared arrays are reusable when thresholds/ROIs are
revised; changing preparation settings requires a new preparation. No file paths,
frame boundaries or particle assignments from the trial dataset are embedded.
"""
import warnings
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares, linear_sum_assignment
from skimage.feature import peak_local_max, blob_log
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

try:
    from .dwell_detection import register, episodes, level_for_detection
except ImportError:  # Same helper is copied into the isolated analysis workspace.
    from dwell_detection import register, episodes, level_for_detection


def prepare(frames, *, plane=True, block_frames=10, baseline_percentile=25., check=lambda: None):
    """Compute motion and temporal/compact responses once, before AI review."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) < 2 or min(frames.shape[1:]) < 12 or not np.isfinite(frames).all():
        raise ValueError('Dwell preparation needs finite images, at least 2 frames and 12 pixels per axis.')
    if type(block_frames) is not int or block_frames < 1 or not 0 <= baseline_percentile <= 100:
        raise ValueError('Invalid preparation parameters.')
    block_frames = min(block_frames, len(frames))
    a = level_for_detection(frames, check) if plane else frames.astype('f4').copy()
    n, h, w = a.shape
    cap = float(np.percentile(a, 70))
    def texture(im):
        check()
        b = np.minimum(ndi.gaussian_filter(im, .8), cap)
        return b-ndi.gaussian_filter(b, 4)
    refs = np.array([texture(np.median(a[k:k+block_frames], axis=0)) for k in range(0, n, block_frames)])
    edges = []
    for j in range(1, len(refs)):
        for gap in (1, 2, 4):
            if j >= gap:
                check(); delta, score = register(refs[j-gap], refs[j], min(12, 3+gap*2))
                edges.append((j-gap, j, delta, score))
    trajectory = np.zeros((len(refs), 2)); graph_error = np.empty(0)
    if edges:
        design = np.zeros((len(edges), len(refs)-1))
        delta = np.array([e[2] for e in edges]); weights = np.maximum(.05, [e[3] for e in edges])**2
        for k, (i, j, _, _) in enumerate(edges):
            design[k, j-1] = 1
            if i: design[k, i-1] = -1
        for axis in (0, 1):
            check()
            initial = np.linalg.lstsq(design*weights[:, None], delta[:, axis]*weights, rcond=None)[0]
            trajectory[1:, axis] = least_squares(lambda z: (design@z-delta[:, axis])*weights,
                                                initial, loss='soft_l1', f_scale=.3).x
        graph_error = np.linalg.norm(design@trajectory[1:]-delta, axis=1)
    local = [register(refs[i//block_frames], texture(im), 3) for i, im in enumerate(a)]
    shifts = np.array([trajectory[i//block_frames]+d for i, (d, _) in enumerate(local)])
    quality = np.array([q for _, q in local])
    lo = np.floor(shifts.min(0)).astype(int)-2; hi = np.ceil(shifts.max(0)).astype(int)+2
    H, W = np.array([h, w])+hi-lo
    if n*int(H)*int(W) > 48_000_000:
        raise ValueError('Expanded dwell canvas is too large; split unrelated fields or use the original detector.')
    aligned = np.full((n, H, W), np.nan, dtype='f4')
    for i, (im, shift) in enumerate(zip(a, shifts)):
        check(); origin = shift-lo; base = np.floor(origin).astype(int)
        small = ndi.shift(im, origin-base, order=1, mode='constant', cval=np.nan, prefilter=False)
        aligned[i, base[0]:base[0]+h, base[1]:base[1]+w] = small
    check()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        baseline = np.nanpercentile(aligned, baseline_percentile, axis=0)
        residual = aligned-baseline
        residual -= np.nanmedian(residual, axis=2)[:, :, None]
    residual = np.nan_to_num(residual)
    response = ndi.gaussian_filter(residual, (0, 1, 1))-ndi.gaussian_filter(residual, (0, 5, 5))
    finite = np.isfinite(aligned)
    support = ndi.minimum_filter(finite.astype('u1'), size=(1, 9, 9), mode='constant') > 0
    filled = np.nan_to_num(aligned)
    compact = ndi.gaussian_filter(filled, (0, 1, 1))-ndi.gaussian_filter(filled, (0, 3, 3))
    check(); samples = np.diff(response, axis=0)[support[1:] & support[:-1]]
    noise = float(1.4826*np.median(np.abs(samples-np.median(samples)))/np.sqrt(2)) if samples.size else 0.
    # Spatial evidence must survive a temporal baseline that removes persistently
    # occupied particles. This is analysis-only; the displayed heights stay intact.
    spatial = ndi.gaussian_filter(a, (0, 2, 2))-ndi.gaussian_filter(a, (0, 10, 10))
    return dict(version=np.array(3), shape=np.array(a.shape), response=response, compact=compact,
                spatial=spatial, leveled=a, spatial_sigma_px=np.array(2.), spatial_background_sigma_px=np.array(10.),
                support=support, shifts=shifts, origin_yx=lo, quality=quality, noise_nm=np.array(noise),
                graph_error_px=graph_error, plane=np.array(plane), block_frames=np.array(block_frames),
                baseline_percentile=np.array(baseline_percentile), texture_cap_nm=np.array(cap))


def with_spatial_settings(prepared, *, sigma_px=2., background_sigma_px=10., check=lambda: None):
    """Rebuild detection maps only; never refit drift or alter height traces.

    Returns a new mapping. The captured source and original prepared cache remain
    unchanged. Legacy caches need a new preparation to supply leveled images.
    """
    if (not np.isfinite([sigma_px, background_sigma_px]).all()
            or not 0 < sigma_px < background_sigma_px):
        raise ValueError('Spatial scales need 0 < sigma < background sigma.')
    if 'leveled' not in prepared:
        raise ValueError('This legacy cache has no leveled images; prepare a separate new cache.')
    maps = []
    for image in prepared['leveled']:
        check()
        maps.append(ndi.gaussian_filter(image, sigma_px)-ndi.gaussian_filter(image, background_sigma_px))
    return dict(prepared, spatial=np.asarray(maps), spatial_sigma_px=np.array(sigma_px),
                spatial_background_sigma_px=np.array(background_sigma_px))


def spatial_observations(prepared, *, prominence=.7, snr=1.6, method='watershed',
                         log_min_sigma=1., log_max_sigma=4., check=lambda: None):
    """Measure whole bright features in each frame, independently of event calls.

    Watershed seeds use prominence, not every scan-line maximum. Weak and edge
    footprints are retained with flags; no presence observation is a dwell event.
    """
    if (method not in ('watershed', 'log_watershed') or
            not np.isfinite([prominence, snr, log_min_sigma, log_max_sigma]).all() or
            min(prominence, snr, log_min_sigma) <= 0 or log_max_sigma < log_min_sigma):
        raise ValueError('Invalid spatial detector settings.')
    if 'spatial' not in prepared: return []  # Legacy prepared caches remain usable.
    output = []
    for image in prepared['spatial']:
        check()
        noise = float(1.4826*np.median(np.abs(image-np.median(image))))
        if noise <= 1e-8 or np.max(image) <= 1e-8:
            output.append([]); continue
        if method == 'watershed':
            markers, _ = ndi.label(h_maxima(image, max(noise*prominence, 1e-8)) & (image > noise))
        else:
            # Scale-space centers can separate peaks joined by broad shoulders.
            # They remain proposals: scan streaks may also generate blob peaks.
            blobs = blob_log(image, min_sigma=log_min_sigma, max_sigma=log_max_sigma,
                             num_sigma=6, threshold=max(noise*prominence, 1e-8), overlap=.5,
                             exclude_border=False)
            markers = np.zeros(image.shape, dtype=int)
            for ident, (y, x, _) in enumerate(blobs, 1):
                y, x = int(round(y)), int(round(x))
                if image[y, x] > noise: markers[y, x] = ident
        labels = watershed(-image, markers, mask=image > noise*.65)
        observations = []
        for label, slices in enumerate(ndi.find_objects(labels), 1):
            if slices is None: continue
            ys, xs = slices
            yy, xx = np.nonzero(labels[slices] == label); yy += ys.start; xx += xs.start
            values = image[yy, xx]
            if len(xx) < 8 or float(values.max()) < noise*snr: continue
            weights = np.maximum(values, 0)
            cx, cy = float(np.average(xx, weights=weights)), float(np.average(yy, weights=weights))
            # Pixel-center coordinates, with one pixel of visible margin. Clip
            # partial objects instead of inventing extent beyond the field.
            h, w = image.shape
            bounds = [float(max(-.5, xx.min()-1)), float(max(-.5, yy.min()-1)),
                      float(min(w-.5, xx.max()+1)), float(min(h-.5, yy.max()+1))]
            flags = []
            if xs.start == 0 or ys.start == 0 or xs.stop == w or ys.stop == h: flags.append('edge_partial_particle')
            if values.max() < noise*2.5: flags.append('weak_spatial_contrast')
            if (xs.stop-xs.start)/max(1, ys.stop-ys.start) > 3.5: flags.append('scan_streak_or_elongated_particle')
            observations.append(dict(x=cx, y=cy, bounds=bounds, flags=flags,
                                     spatial_peak_nm=float(values.max())))
        output.append(observations)
    return output


def presence_tracks(prepared, observations, separation_px, check, registration_min=.4):
    """Associate measured footprints in drift coordinates; never fill missing detections."""
    tracks = []; shifts = prepared['shifts']; lo = prepared['origin_yx']
    gate = max(4., float(separation_px))
    for frame, detected in enumerate(observations):
        check()
        world = np.array([[p['x'], p['y']] for p in detected]).reshape(-1, 2)+shifts[frame, ::-1]-lo[::-1]
        used = set()
        if tracks and len(world) and prepared['quality'][frame] >= registration_min:
            anchors = np.array([t['anchor'] for t in tracks])
            distances = np.linalg.norm(anchors[:, None, :]-world[None, :, :], axis=2)
            possible = np.flatnonzero(np.min(distances, axis=1) <= gate)
            distances = distances[possible]
            # Dummy columns allow unmatched sites; a far observation must not
            # steal a valid one-to-one correspondence from a close pair.
            costs = np.concatenate((np.where(distances <= gate, distances, gate*4),
                                    np.full((len(possible), len(possible)), gate+1)), axis=1)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if column >= len(world) or distances[row, column] > gate: continue
                t = tracks[possible[row]]; t['observations'][frame] = detected[column]
                # Do not let a noisy long trajectory drag a site onto its neighbor.
                t['world'].append(world[column]); t['anchor'] = np.median(t['world'], axis=0)
                used.add(int(column))
        for index, point in enumerate(world):
            if index in used: continue
            tracks.append(dict(anchor=point.copy(), world=[point], observations={frame: detected[index]}))
            if len(tracks) > 2000: raise ValueError('Too many particle sites; revise spatial detection parameters.')
    return tracks


def _temporal_measure(prepared, *, high_nm=None, low_nm=None, radius_px=2.5, separation_px=6,
            compact_min_nm=.45, density_floor=.003, registration_min=.4, check=lambda: None):
    """Revisit detection/thresholds on measured arrays without repeating motion fitting.

Defaults reproduce the interactive trial's starting method, not universal
thresholds. AI may revise them or replace this method after checking the data.
"""
    n, h, w = map(int, prepared['shape'])
    response = prepared['response']; compact = prepared['compact']; support = prepared['support']
    shifts = prepared['shifts']; lo = prepared['origin_yx']; quality = prepared['quality']
    noise = float(prepared['noise_nm']); high = max(.8, 5*noise) if high_nm is None else float(high_nm)
    low = .6*high if low_nm is None else float(low_nm)
    if (not 0 < low < high or not .5 <= radius_px <= min(h, w)/2 or not 0 <= registration_min <= 1
            or not np.isfinite([high, low, radius_px, compact_min_nm, density_floor]).all()):
        raise ValueError('Invalid dwell measurement parameters.')
    check()
    density = ndi.gaussian_filter(np.mean(np.maximum(response-high, 0)*support, axis=0), 1)
    points = peak_local_max(density, min_distance=max(2, int(separation_px)), threshold_abs=density_floor,
                            exclude_border=7)
    sites = []; events = []; yy, xx = np.indices(response.shape[1:])
    for y, x in sorted(points, key=lambda p: -density[tuple(p)]):
        check()
        if len(sites) >= 2000: raise ValueError('Too many candidate sites; revise detection parameters.')
        disk = (yy-y)**2+(xx-x)**2 <= radius_px**2
        signal = np.percentile(response[:, disk], 80, axis=1)
        raw_peak = np.percentile(compact[:, disk], 90, axis=1)
        positions = np.array([x, y])[None, :]+lo[::-1]-shifts[:, ::-1]
        # Require support for the actual aperture even if AI increases its radius.
        valid = support[:, y, x] & (quality >= registration_min)
        valid &= ((positions[:, 0] >= radius_px+1) & (positions[:, 0] < w-radius_px-1)
                  & (positions[:, 1] >= radius_px+1) & (positions[:, 1] < h-radius_px-1))
        keep = []
        for first, last, bridged in episodes(signal, valid, high, low):
            peak = first+int(np.argmax(signal[first:last+1]))
            if raw_peak[peak] < compact_min_nm: continue
            patch = response[peak, y-6:y+7, max(0, x-9):x+10]
            labels, _ = ndi.label(patch > max(.4, .4*response[peak, y, x]))
            label = labels[6, x-max(0, x-9)]
            coords = np.argwhere(labels == label) if label else np.empty((0, 2))
            flags = []
            if last-first+1 < 3: flags.append('short_event')
            if signal[peak] < max(1.25, high*1.4): flags.append('weak_height_change')
            if bridged: flags.append('one_frame_gap_bridged')
            if len(coords) < 6 or (np.ptp(coords, axis=0)[1]+1)/(np.ptp(coords, axis=0)[0]+1) > 3.5:
                flags.append('scan_streak_or_small_feature')
            if last-first+1 > 100: flags.append('long_occupancy_or_background_change')
            for side, off, on in [('onset', signal[max(0, first-4):first], signal[first:min(first+3, last+1)]),
                                  ('offset', signal[last+1:last+5], signal[max(first, last-2):last+1])]:
                if len(off) < 3 or np.median(off) >= low or off.max() >= high: flags.append('no_stable_low_state_at_'+side)
                if len(off) and np.median(on)-np.median(off) < high*.5: flags.append('gradual_or_uncertain_'+side)
            if first == 0 or not valid[first-1]: flags.append('left_censored')
            if last == n-1 or not valid[last+1]: flags.append('right_censored')
            keep.append((first, last, flags))
        if not keep: continue
        sid = f'S{len(sites)+1:03d}'; bounds = []
        for i, (cx, cy) in enumerate(positions):
            rx = ry = max(4., radius_px)
            if signal[i] >= low and valid[i]:
                lab, _ = ndi.label(response[i, y-6:y+7, x-7:x+8] > max(.35, .35*float(signal[i])))
                component = lab[6, 7]; coords = np.argwhere(lab == component) if component else np.empty((0, 2))
                if len(coords):
                    ry = max(ry, min(6., float(np.max(np.abs(coords[:, 0]-6)))+1))
                    rx = max(rx, min(7., float(np.max(np.abs(coords[:, 1]-7)))+1))
            bounds.append([float(cx-rx), float(cy-ry), float(cx+rx), float(cy+ry)])
        sites.append(dict(id=sid, first_frame=1, last_frame=n, radius_px=float(radius_px),
            display_radius_px=max(4., float(radius_px)), positions_px=positions.tolist(), particle_bounds_px=bounds,
            particle_present=((signal >= low) & (raw_peak > compact_min_nm*.78) & valid).tolist(),
            response_nm=signal.tolist(), valid=valid.tolist(), high_nm=high, low_nm=low))
        for first, last, flags in keep:
            events.append(dict(site=sid, first_frame=first+1, last_frame=last+1,
                status='review_required' if flags else 'clear_candidate', flags=flags,
                left_censored='left_censored' in flags, right_censored='right_censored' in flags))
    events.sort(key=lambda e: (e['first_frame'], e['site']))
    for i, event in enumerate(events, 1): event['id'] = f'E{i:04d}'
    return dict(algorithm='Interactive trial method: drift graph, expanded temporal baseline and compact-peak evidence',
        signal_definition=f'80th percentile of local temporal height change within a {radius_px:g}-pixel disk after row/broad-background removal (nm); compact-peak evidence at event peak.',
        parameters=dict(high_nm=high, low_nm=low, noise_nm=noise, radius_px=radius_px,
            separation_px=separation_px, compact_min_nm=compact_min_nm, density_floor=density_floor,
            registration_min=registration_min, plane_for_detection=bool(prepared['plane']),
            block_frames=int(prepared['block_frames']), baseline_percentile=float(prepared['baseline_percentile'])),
        warnings=['Candidates are not confirmed molecular binding events. Review scan streaks and moving terrain.',
                  'Persistent occupancy and weak/brief events may be missed; inspect input images as well as overlays.',
                  'ROI extent is a local measurement aid, not a guaranteed molecular segmentation.'],
        assessed_frames=list(range(1, n+1)), sites=sites, events=events)



def measure(prepared, *, high_nm=None, low_nm=None, radius_px=2.5, separation_px=6,
            compact_min_nm=.45, density_floor=.003, registration_min=.4,
            presence_snr=1.6, presence_prominence=.7, presence_method='watershed',
            log_min_sigma=1., log_max_sigma=4., check=lambda: None):
    """Detect visible footprints without making their changing shape a dwell signal.

    Height apertures stay fixed in drift-registered coordinates. Particle centers
    and extents are separate observations: moving an aperture along a changing
    bright structure can create false on/off transitions. Both geometries are
    retained in JSON and the selected height aperture is drawn in review.
    """
    if (not np.isfinite([presence_snr, presence_prominence]).all()
            or min(presence_snr, presence_prominence) <= 0):
        raise ValueError('Invalid spatial detection parameters.')
    result = _temporal_measure(prepared, high_nm=high_nm, low_nm=low_nm, radius_px=radius_px,
        separation_px=separation_px, compact_min_nm=compact_min_nm, density_floor=density_floor,
        registration_min=registration_min, check=check)
    observations = spatial_observations(prepared, snr=presence_snr, prominence=presence_prominence,
        method=presence_method, log_min_sigma=log_min_sigma, log_max_sigma=log_max_sigma, check=check)
    if not observations: return result  # Existing prepared caches preserve their behavior.
    n, h, w = map(int, prepared['shape']); sites = result['sites']
    for site in sites:
        site['measurement_positions_px'] = [xy[:] for xy in site['positions_px']]
        site['particle_flags'] = [[] for _ in range(n)]
        site['presence_source'] = 'temporal_scout_with_spatial_footprints'
    remaining = []
    for frame, detected in enumerate(observations):
        check(); used = set()
        if sites and detected:
            anchors = np.asarray([s['measurement_positions_px'][frame] for s in sites])
            centers = np.asarray([[p['x'], p['y']] for p in detected])
            bounds = np.asarray([p['bounds'] for p in detected])
            inside = np.all((anchors[:, None, :] >= bounds[None, :, :2])
                            & (anchors[:, None, :] <= bounds[None, :, 2:]), axis=2)
            distances = np.linalg.norm(anchors[:, None, :]-centers[None, :, :], axis=2)
            costs = np.concatenate((np.where(inside, distances, 1e6), np.full((len(sites), len(sites)), 1e4)), axis=1)
            rows, columns = linear_sum_assignment(costs)
            assigned = set()
            for row, column in zip(rows, columns):
                if column >= len(detected) or not inside[row, column]: continue
                assigned.add(int(row))
                point = detected[column]; site = sites[row]; used.add(int(column))
                site['positions_px'][frame] = [point['x'], point['y']]
                site['particle_bounds_px'][frame] = point['bounds']
                site['particle_present'][frame] = True
                site['particle_flags'][frame] = point['flags']
            # Multiple temporal apertures may lie inside one broad particle.
            # Preserve their measured traces but do not draw tiny false particles
            # on its flanks. The shared footprint is explicitly ambiguous.
            for row in range(len(sites)):
                candidates = np.flatnonzero(inside[row])
                if row in assigned or not len(candidates): continue
                column = min(candidates, key=lambda c: distances[row, c])
                point = detected[column]; site = sites[row]; used.add(int(column))
                site['positions_px'][frame] = [point['x'], point['y']]
                site['particle_bounds_px'][frame] = point['bounds']
                site['particle_present'][frame] = True
                site['particle_flags'][frame] = point['flags']+['shared_particle_footprint']
                for other in sites:
                    if other['particle_bounds_px'][frame] == point['bounds']:
                        other['particle_flags'][frame] = list(dict.fromkeys(other['particle_flags'][frame]+['shared_particle_footprint']))
        remaining.append([p for i, p in enumerate(detected) if i not in used])
    tracks = presence_tracks(prepared, remaining, separation_px, check, registration_min)
    shifts = prepared['shifts']; lo = prepared['origin_yx']
    yy, xx = np.indices(prepared['response'].shape[1:])
    for track in tracks:
        check()
        if len(sites) >= 2000: raise ValueError('Too many particle sites; revise spatial detection parameters.')
        x, y = np.rint(track['anchor']).astype(int)
        aperture = np.array([x, y])[None, :]+lo[::-1]-shifts[:, ::-1]
        positions = aperture.copy(); bounds = np.column_stack((positions-4, positions+4))
        present = np.zeros(n, bool); flags = [[] for _ in range(n)]
        for i, point in track['observations'].items():
            positions[i] = [point['x'], point['y']]; bounds[i] = point['bounds']; present[i] = True; flags[i] = point['flags']
        disk = (xx-x)**2+(yy-y)**2 <= radius_px**2
        signal = np.percentile(prepared['response'][:, disk], 80, axis=1)
        valid = prepared['support'][:, y, x] & (prepared['quality'] >= registration_min)
        valid &= ((aperture[:, 0] >= radius_px+1) & (aperture[:, 0] < w-radius_px-1)
                  & (aperture[:, 1] >= radius_px+1) & (aperture[:, 1] < h-radius_px-1))
        valid &= np.array(['edge_partial_particle' not in f for f in flags])
        sites.append(dict(id=f'S{len(sites)+1:03d}', first_frame=1, last_frame=n,
            positions_px=positions.tolist(), measurement_positions_px=aperture.tolist(),
            particle_bounds_px=bounds.tolist(), particle_present=present.tolist(), particle_flags=flags,
            presence_source='per_frame_spatial', radius_px=float(radius_px), display_radius_px=max(4., float(radius_px)),
            response_nm=signal.tolist(), valid=valid.tolist(), high_nm=None, low_nm=None))
    result['algorithm'] = 'Spatial particle footprints with drift-registered temporal event evidence'
    result['signal_definition'] = (f'80th percentile of temporal height change in a {radius_px:g}-pixel disk fixed '
        'in drift-registered coordinates (nm). Cross/box: measured particle center/extent. '
        'Circle on the selected site: height aperture. Spatial-only sites have no inferred dwell event.')
    result['parameters'].update(presence_snr=presence_snr, presence_prominence=presence_prominence,
                                presence_method=presence_method, log_min_sigma=log_min_sigma,
                                log_max_sigma=log_max_sigma,
                                spatial_sigma_px=float(prepared.get('spatial_sigma_px', 2.)),
                                spatial_background_sigma_px=float(prepared.get('spatial_background_sigma_px', 10.)),
                                spatial_frame_counts=[len(p) for p in observations])
    result['warnings'] = ['Candidates are not confirmed molecular binding events. Review scan streaks and moving terrain.',
        'Visible particles without a supported transition keep their ROI; presence alone does not establish binding/unbinding.',
        'ROI extent is a local measurement aid, not a guaranteed molecular segmentation.']
    return result



def save_prepared(prepared, path):
    np.savez_compressed(path, **prepared)


def load_prepared(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
