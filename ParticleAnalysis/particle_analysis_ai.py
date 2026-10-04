"""AI segmentation and movie identities, with host-measured geometry and heights.

AI-authored code runs only through CodexAnalysisClient's sandbox. The GUI never
imports it. Measurements use the captured numerical image, not detection filters
or display pixels. No ASD writing.
"""
from ai_recovery import bounded_analysis, current_budget, is_timeout_message
from copy import deepcopy
from analysis_input import (FrameInput, capture_frame, capture_frames, source_unchanged,
                            MAX_PIXELS, MAX_SESSION_PIXELS)
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from background_auto import check_cancel
from codex_app_server import CodexError
from analysis_runtime import (open_host_output, write_host_text, replay_timeout, isolated_replay, REPLAY_RULE, digest, read_workspace_file, worker_command,
                            result_schema, check_reply, review_measurements)

FORMAT = 'pynud-particle-frame-v1'
MAX_PARTICLES = 10000
COLORS = {'stable': '#26f5ad', 'uncertain': '#ff59d3', 'edge': '#60c5ff'}


def load_array(root, name, shape, kind):
    """Validate NPY headers and size before loading; never accept pickle or links."""
    path = Path(root) / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > np.prod(shape)*8+65536:
        raise ValueError('Invalid or oversized ' + name)
    content = path.read_bytes(); stream = io.BytesIO(content)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0): dims, order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0): dims, order, dtype = np.lib.format.read_array_header_2_0(stream)
    else: raise ValueError('Unsupported NPY version for ' + name)
    if tuple(dims) != tuple(shape) or dtype.kind not in kind or dtype.hasobject:
        raise ValueError('Wrong shape or dtype for ' + name)
    if len(content)-stream.tell() != int(np.prod(shape))*dtype.itemsize:
        raise ValueError('Incomplete NPY array: ' + name)
    return np.load(io.BytesIO(content), allow_pickle=False)


def validate_segmentation(root, snapshot, expected_hash):
    raw = json.loads(read_workspace_file(root, 'result.json', 8*1024*1024),
                     parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
    if (not isinstance(raw, dict) or raw.get('format') != FORMAT or raw.get('input_sha256') != expected_hash
            or not isinstance(raw.get('algorithm'), str) or not raw['algorithm'].strip()
            or not isinstance(raw.get('parameters'), dict) or not isinstance(raw.get('particles'), list)
            or not isinstance(raw.get('warnings'), list) or any(not isinstance(w, str) for w in raw['warnings'])):
        raise ValueError('Invalid segmentation result contract.')
    labels = load_array(root, 'labels.npy', snapshot.image.shape, 'iu')
    background = load_array(root, 'background.npy', snapshot.image.shape, 'b')
    validate_masks(labels, background, raw['particles'])
    return labels.astype(np.int32), background, raw


def validate_masks(labels, background, records):
    if labels.ndim != 2 or labels.dtype.kind not in 'iu' or background.shape != labels.shape or background.dtype.kind != 'b':
        raise ValueError('Invalid label/background arrays.')
    if labels.min() < 0 or labels.max() > MAX_PARTICLES:
        raise ValueError('Segmentation label IDs must be between 0 and 10000.')
    labels = labels.astype(np.int32)
    ids = set(map(int, np.unique(labels))) - {0}
    if len(records) != len(ids): raise ValueError('Every segmented region needs one particle record.')
    seen = set()
    for p in records:
        if (not isinstance(p, dict) or type(p.get('id')) is not int or p['id'] not in ids or p['id'] in seen
                or type(p.get('uncertain')) is not bool or not isinstance(p.get('reason'), str)):
            raise ValueError('Invalid or duplicate particle record.')
        seen.add(p['id'])
        if type(p.get('identity_uncertain', False)) is not bool:
            raise ValueError('Invalid identity uncertainty flag.')
        if type(p.get('near_edge', False)) is not bool:
            raise ValueError('Invalid near-edge review flag.')
        if 'detection_changed' in p and type(p['detection_changed']) is not bool:
            raise ValueError('Invalid detection change flag.')
        if 'source_local_ids' in p:
            local=p['source_local_ids']
            if (not isinstance(local,list) or any(type(i) is not int or not 1<=i<=MAX_PARTICLES for i in local)
                    or len(local)!=len(set(local))):
                raise ValueError('Invalid source detection IDs.')
    if np.any(background & (labels > 0)):
        raise ValueError('Background pixels cannot also belong to particles.')
    for region in measure.regionprops(labels):
        if ndi.label(region.image, structure=np.ones((3, 3)))[1] != 1:
            raise ValueError(f'Particle {region.label}: disconnected components need separate IDs.')


def measure_particles(snapshot, labels, background, raw, cancelled=lambda: False):
    """Geometry calibrated in X and Y; heights from unmodified captured input.

    Local baseline: median of agent-approved background within a nearby annulus,
    excluding ALL particle labels. With fewer than 8 such pixels, relative height
    and signed volume are unavailable. No zero/global baseline is silently used.
    """
    image = snapshot.image; dx, dy = snapshot.pixel_size; h, w = image.shape
    by_id = {p['id']: p for p in raw['particles']}; records = []
    for region in measure.regionprops(labels):
        check_cancel(cancelled)
        ident = int(region.label); top, left, bottom, right = region.bbox
        mask = region.image; vals = image[top:bottom, left:right][mask]
        area = float(region.area)*dx*dy
        diameter = math.sqrt(4*area/math.pi)
        contours = measure.find_contours(np.pad(mask.astype(float), 1), .5)
        perimeter = 0.; polygons = []
        for contour in contours:
            xy = contour[:, ::-1] - 1 + [left, top]
            perimeter += float(np.linalg.norm(np.diff(xy, axis=0)*[dx, dy], axis=1).sum())
            polygons.append(xy.tolist())
        physical = region.coords[:, ::-1]*[dx, dy]
        cov = np.cov(physical.T, bias=True) if len(physical)>1 else np.zeros((2, 2))
        eigen, vec = np.linalg.eigh(cov); eigen = np.maximum(eigen, 0)
        minor, major = 4*np.sqrt(eigen)
        orientation = math.degrees(math.atan2(vec[1, 1], vec[0, 1])) % 180
        # Local field radius follows object size in calibrated units.
        margin = max(2*max(dx, dy), .5*diameter)
        mx, my = math.ceil(margin/dx), math.ceil(margin/dy)
        y0, y1 = max(0, top-my), min(h, bottom+my)
        x0, x1 = max(0, left-mx), min(w, right+mx)
        sub = labels[y0:y1, x0:x1]
        distances = ndi.distance_transform_edt(sub != ident, sampling=(dy, dx))
        ring = ((distances > 0) & (distances <= margin) & background[y0:y1, x0:x1] & (sub == 0))
        count = int(ring.sum())
        baseline = float(np.median(image[y0:y1, x0:x1][ring])) if count >= 8 else None
        touches_edge = top == 0 or left == 0 or bottom == h or right == w
        info = by_id[ident]
        records.append(dict(id=ident, uncertain=info['uncertain'] or info.get('identity_uncertain',False),
            identity_uncertain=info.get('identity_uncertain',False),
            edge=touches_edge or info.get('near_edge',False), near_edge=info.get('near_edge',False), reason=info['reason'],
            x_px=float(region.centroid[1]), y_px=float(region.centroid[0]),
            x_nm=float((region.centroid[1]+.5)*dx), y_nm=float((region.centroid[0]+.5)*dy),
            area_nm2=area, perimeter_nm=perimeter, circularity=min(1., 4*math.pi*area/perimeter**2) if perimeter else None,
            equivalent_diameter_nm=diameter, major_axis_nm=float(major), minor_axis_nm=float(minor),
            aspect_ratio=float(major/minor) if minor > 0 else None, orientation_deg=float(orientation),
            min_z_nm=float(vals.min()), max_z_nm=float(vals.max()), mean_z_nm=float(vals.mean()),
            std_z_nm=float(vals.std()), background_nm=baseline, background_pixels=count,
            max_above_bg_nm=float(vals.max()-baseline) if baseline is not None else None,
            mean_above_bg_nm=float(vals.mean()-baseline) if baseline is not None else None,
            volume_above_bg_nm3=float(np.sum(vals-baseline)*dx*dy) if baseline is not None else None,
            bbox_xyxy=[left, top, right, bottom], polygons_xy=polygons))
        for key in ('source_local_ids','detection_changed'):
            if key in info:records[-1][key]=deepcopy(info[key])
    return records


def measure_reusing_previous(snapshot, labels, background, raw, previous=None, cancelled=lambda: False):
    """ID/annotation-only repairs do not change calibrated geometry or heights."""
    check_cancel(cancelled)
    if (previous is not None and snapshot.pixel_size == previous['snapshot'].pixel_size
            and np.array_equal(snapshot.image, previous['snapshot'].image)
            and np.array_equal(background, previous['background'])
            and np.array_equal(labels > 0, previous['labels'] > 0)):
        mask = labels > 0
        pairs = np.unique(np.column_stack((previous['labels'][mask], labels[mask])), axis=0)
        mapping = {int(a): int(b) for a, b in pairs}
        if (len(pairs) == len(mapping) == len(previous['particles'])
                and len(set(mapping.values())) == len(mapping)):
            notes = {p['id']: p for p in raw['particles']}
            rows = []
            for prior in previous['particles']:
                check_cancel(cancelled)
                row = deepcopy(prior); row['id'] = mapping[prior['id']]; note = notes[row['id']]
                row.update(uncertain=note['uncertain'] or note.get('identity_uncertain', False),
                    identity_uncertain=note.get('identity_uncertain', False),
                    near_edge=note.get('near_edge', False), reason=note['reason'])
                left, top, right, bottom = row['bbox_xyxy']
                row['edge'] = (left == 0 or top == 0 or right == labels.shape[1]
                               or bottom == labels.shape[0] or row['near_edge'])
                for key in ('source_local_ids', 'detection_changed'):
                    row.pop(key, None)
                    if key in note: row[key] = deepcopy(note[key])
                rows.append(row)
            return sorted(rows, key=lambda p: p['id']), True
    return measure_particles(snapshot, labels, background, raw, cancelled), False


def save_measured_checkpoint(root, version, results, immutable, *, movie=False):
    """Keep only host-validated masks as explicit replay inputs, never stale outputs."""
    from main_view_data import pack_frame
    labels = np.stack([r['labels'] for r in results]) if movie else results[0]['labels']
    background = np.stack([r['background'] for r in results]) if movie else results[0]['background']
    prefix = f'checkpoint_{version}'
    with open_host_output(root, prefix+'_labels.npy') as stream: np.save(stream, labels, allow_pickle=False)
    with open_host_output(root, prefix+'_background.npy') as stream: np.save(stream, background, allow_pickle=False)
    raw = [r['audit']['raw_result'] for r in results]
    write_host_text(root, prefix+'.json', json.dumps(raw if movie else raw[0], allow_nan=False))
    for suffix in ('_labels.npy', '_background.npy', '.json'):
        immutable[prefix+suffix] = digest(root/(prefix+suffix))
    # Each frame carries only its own checkpoint data in portable JSON exports.
    for result in results:
        result['audit'].setdefault('checkpoint_inputs', {})[str(version)] = dict(
            labels=pack_frame(result['labels']), background=pack_frame(result['background']),
            raw_result=deepcopy(result['audit']['raw_result']))


def particle_color(record):
    return COLORS['edge' if record['edge'] else 'uncertain' if record['uncertain'] else 'stable']


def render(snapshot, records=(), selected=None, show_ids=True, highlight=None, long_edge=1000):
    """Flip numerical rows and overlays together once, as in the main view."""
    from PIL import Image, ImageDraw, ImageFont
    from main_view_data import display_tone_pixels
    a = snapshot.image; h, w = a.shape; dx, dy = snapshot.pixel_size
    import cv2
    pixels = display_tone_pixels(a, snapshot.display)
    ratio = w*dx/(h*dy)
    size = (long_edge, max(1, round(long_edge/ratio))) if ratio >= 1 else (max(1, round(long_edge*ratio)), long_edge)
    # Main view resizes the gray image before vertical flip and color LUT.
    out = Image.fromarray(snapshot.palette[np.flipud(cv2.resize(pixels, size))])
    d = ImageDraw.Draw(out); sx, sy = size[0]/w, size[1]/h
    font = ImageFont.load_default()
    for name in ('DejaVuSans.ttf', '/System/Library/Fonts/Helvetica.ttc', 'arial.ttf'):
        try:
            font = ImageFont.truetype(name, max(14, round(long_edge*.032)))
            break
        except OSError: pass
    for p in records:
        color = particle_color(p); chosen = selected is None or p['id'] in selected
        # Unselected candidates must remain visible when the GUI scales down.
        width = 5 if p['id'] == highlight else 3
        for poly in p['polygons_xy']:
            pts = [((x+.5)*sx, (h-y-.5)*sy) for x, y in poly]
            d.line(pts, fill=color, width=width)
        x, y = (p['x_px']+.5)*sx, (h-p['y_px']-.5)*sy
        d.line((x-3,y,x+3,y), fill=color, width=1); d.line((x,y-3,x,y+3), fill=color, width=1)
        if p['uncertain'] and p['edge']:
            d.ellipse((x-4,y-4,x+4,y+4), outline=COLORS['uncertain'], width=2)
        if show_ids:
            name = str(p['id']) + ('' if chosen else '×')
            bounds = d.textbbox((0,0),name,font=font)
            d.text((max(1,min(size[0]-bounds[2]-2,x+4)),max(1,min(size[1]-bounds[3]-2,y-bounds[3]))), name, font=font,
                   fill=color, stroke_width=1, stroke_fill='black')
    return out


SUPPORT_CODE = '''from pathlib import Path
import json
import numpy as np
from functools import lru_cache

def load():
    return np.load('frame.npy',allow_pickle=False),json.loads(Path('metadata.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=4)
def load_checkpoint(version=1):
    prefix=f'checkpoint_{int(version)}'
    return (np.load(prefix+'_labels.npy',mmap_mode='r',allow_pickle=False),
            np.load(prefix+'_background.npy',mmap_mode='r',allow_pickle=False),
            json.loads(Path(prefix+'.json').read_text(encoding='utf-8')))

def save_checkpoint(version=1):
    labels,background,raw=load_checkpoint(version)
    save(labels,background,raw['particles'],raw['algorithm'],raw['parameters'],raw['warnings'])

def save(labels, background, particles, algorithm, parameters=None, warnings=None):
    _,meta=load()
    np.save('labels.npy',np.asarray(labels,dtype=np.int32),allow_pickle=False)
    np.save('background.npy',np.asarray(background,dtype=bool),allow_pickle=False)
    def scalar(v):
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v).__name__)
    Path('result.json').write_text(json.dumps(dict(format='pynud-particle-frame-v1',
        input_sha256=meta['input_sha256'],algorithm=algorithm,parameters=parameters or {},
        warnings=warnings or [],particles=particles),default=scalar,allow_nan=False))
'''


NORMAL_FRAME_SUPPORT = '''
def load_detection():
    return (np.load('normal_labels.npy',allow_pickle=False),
            np.load('normal_background.npy',allow_pickle=False),
            json.loads(Path('normal_detection.json').read_text(encoding='utf-8')))

def save_detection(annotations=None, background=None, parameters=None, warnings=None):
    labels,bg,meta=load_detection()
    annotations=annotations or {};particles=meta['particles']
    for p in particles:
        note=annotations.get(p['id'],annotations.get(str(p['id']),{}))
        p['uncertain']=bool(p['uncertain'] or note.get('uncertain',False))
        p['reason']=str(note.get('reason',p['reason']))
    params=dict(meta['parameters']);params.update(parameters or {});params['detection_revisions']=[]
    save(labels,bg if background is None else background,particles,
         'Normal detection + AI review',params,warnings)
'''


@bounded_analysis
def run_analysis(snapshot, client, workspace, instructions='', progress=lambda p,m: None,
                 cancelled=lambda: False, time_limit=900, max_rounds=2, initial_detection=None, refinement=None):
    root = Path(workspace).resolve(); root.mkdir(parents=True, exist_ok=True)
    if type(max_rounds) is not int or not 1<=max_rounds<=12: raise ValueError('AI rounds must be between 1 and 12.')
    budget = current_budget()
    def check(check_time=True):
        check_cancel(cancelled); source_unchanged(snapshot)
        budget.remaining(check_time)
    percent=0
    def report(n, msg):
        nonlocal percent
        check();percent=n;progress(n,msg)
    timings=[]
    def timed(stage, operation):
        started=time.monotonic()
        try:
            result=operation();budget.touch();return result
        finally:
            seconds=round(time.monotonic()-started,3)
            timings.append(dict(stage=stage,seconds=seconds))
            progress(percent,f'{stage}: {seconds:.1f} s.')
    check(); np.save(root/'frame.npy', snapshot.image, allow_pickle=False)
    input_hash = digest(root/'frame.npy')
    meta = dict(format=FORMAT, input_sha256=input_hash, shape=list(snapshot.image.shape),
        pixel_size_nm=list(snapshot.pixel_size), height_unit='nm', frame_0based=snapshot.metadata['frame_0based'],
        coordinates='labels[row,column] aligns directly with frame[row,column]. Do not flip labels. '
            'Preview is flipped vertically by the host; array row 0 is at the BOTTOM of preview.',
        input='One main-window processed numerical frame, with background and filters already applied. '
            'Do not double-correct input or replace its measured heights.')
    (root/'metadata.json').write_text(json.dumps(meta, allow_nan=False))
    seeded=initial_detection is not None
    support=SUPPORT_CODE+(NORMAL_FRAME_SUPPORT if seeded else '')
    (root/'analysis_support.py').write_text(support)
    render(snapshot, long_edge=1000).save(root/'input.png')
    seed_files=[];seed_audit=dict(detection_start_mode='default_detection' if seeded else 'ai_from_scratch')
    if seeded:
        from main_view_data import pack_frame
        seed=initial_detection;before=seed['snapshot']
        if (before.metadata['frame_0based']!=snapshot.metadata['frame_0based'] or before.pixel_size!=snapshot.pixel_size
                or not np.array_equal(before.image,snapshot.image) or seed['labels'].shape!=snapshot.image.shape):
            raise ValueError('Normal detection does not match the captured main frame.')
        validate_masks(seed['labels'],seed['background'],seed['particles'])
        np.save(root/'normal_labels.npy',seed['labels'],allow_pickle=False)
        np.save(root/'normal_background.npy',seed['background'],allow_pickle=False)
        normal_raw=dict(format=FORMAT,input_sha256=input_hash,algorithm='Normal detection + AI review',
            parameters={**deepcopy(seed['parameters']),'detection_revisions':[]},warnings=list(seed['warnings']),
            particles=[dict(id=p['id'],uncertain=p['uncertain'],near_edge=p['edge'],reason=p['reason']) for p in seed['particles']])
        measured=[{k:p[k] for k in ('id','x_px','y_px','bbox_xyxy','area_nm2','equivalent_diameter_nm',
            'major_axis_nm','minor_axis_nm','mean_z_nm','max_z_nm','std_z_nm','background_nm','edge')} for p in seed['particles']]
        (root/'normal_detection.json').write_text(json.dumps(dict(normal_raw,
            measured_particles=measured,selected_local_ids=sorted(selected_ids(seed))),allow_nan=False))
        render(snapshot,seed['particles'],long_edge=1200).save(root/'normal_preview.png')
        seed_files=['normal_labels.npy','normal_background.npy','normal_detection.json','normal_preview.png']
        seed_audit.update(normal_detection=dict(labels=pack_frame(seed['labels']),background=pack_frame(seed['background']),
            particles=deepcopy(seed['particles']),selected_local_ids=sorted(selected_ids(seed)),
            parameters=deepcopy(seed['parameters']),origin=seed.get('seed_origin','supplied normal detection')))
    immutable = {name:digest(root/name) for name in ['frame.npy','metadata.json','analysis_support.py','input.png']+seed_files}
    if refinement is not None:
        prepare_refinement_inputs(root, refinement, [seed], immutable)
    def verify(check_time=True):
        check(check_time)
        if any((root/n).is_symlink() or digest(root/n)!=value for n,value in immutable.items()):
            raise ValueError('AI changed its supplied input; result rejected.')
    command = worker_command(root)
    contract = dict(format=FORMAT, input_sha256=input_hash, algorithm='Actual method and reasoning',
        parameters={'actual_choices':'record actual choices'}, warnings=[],
        particles=[dict(id=1, uncertain=False, reason='Explain the measured candidate and any uncertainty')])
    payload = dict(task='Detect and segment particles in THIS single AFM height image. No tracking. '
        'Choose and test a data-dependent algorithm, write analysis.py and run it with the supplied command. '
        'Inspect the data and overlays; do not restrict yourself to a preset detector. '
        'Make one focused analysis pass and inspect its outlines. Avoid repeated full-image sweeps or polishing; '
        'report remaining ambiguous regions for the measured-result review.',
        runtime_command=command, libraries=['numpy','scipy','skimage','cv2','PIL'],
        helpers='from analysis_support import load, save. load() returns frame,metadata. '
            'save(labels,background,particles,algorithm,parameters,warnings) writes the three output files.',
        output_contract=contract, numerical_metadata=meta, user_instructions=instructions,
        requirements=[
            'Return labels.npy: integer segmentation, same array shape, 0 background, positive unique IDs <=10000. '
            'Each particle is ONE connected region; touching particles may share a border but never pixels. '
            'Return background.npy: boolean same shape, TRUE only for supported substrate/background pixels outside ALL particles. '
            'It may be all False if a defensible baseline cannot be determined. Never invent a background height.',
            'Every nonzero label needs id, uncertain (boolean), reason in result.json. Edge detection and geometry are measured by the host. '
            'Keep partially visible edge particles and plausible weak candidates for HUMAN review, flagged appropriately. '
            'Do not report confidence probabilities without calibrated evidence.',
            'Measure, never hardcode particle coordinates, label masks, counts, or a desired outcome. '
            'Adapt smoothing and scale range to these data. Start with one well-supported method; compare alternatives only '
            'where observed ambiguity or parameter sensitivity warrants it, not as an obligatory whole-frame search. Inspect weak/small particles, '
            'touching particles, false splits and scan artifacts. Changes in particle height alone do not imply different species.',
            'Boundary definition must be physically defensible and recorded. Do not fill entire watershed basins with substrate. '
            'Use local-background or other supported boundary evidence; note parameter sensitivity. '
            'Internally smoothed images are for detection only; host measurement uses the original supplied numerical image.',
            'Coordinates MUST match NUMPY indexing of frame.npy. Do not flip rows to match the preview. '
            'Use input.png only for visual context. Do not infer nm heights from its colors.',
            'Preserve all input/helper files. No network, packages, unrelated files, or application source edits. '
            'Use only the workspace. analysis.py must regenerate all outputs from input. Do not leave background processes running.'])
    seed_rule=('Inspect the input data first and use supplied normal masks and measurements as a reference, not a mandatory detector. '
        'Preserve correct contours and plausible weak/edge candidates for human review. Choose methods and particle scales for THIS data. '
        'If failures are local, repair only those regions. If the method systematically merges particles, misses a size class, '
        'or follows scan artifacts, you may replace the detector across the affected frame. Compare methods on representative '
        'regions first and record parameters.method_reassessment with the observed failure, compared methods and chosen method. '
        'Reuse correct measurements; do not repeat equivalent calculations or perform obligatory parameter sweeps. '
        'Each changed, added or removed particle must be '
        'covered by parameters.detection_revisions=[{bbox_yx:[top,left,bottom,right],reason:"measured defect and repair"}]. '
        'Use half-open NumPy coordinates covering the entire old/new particle. Outside these boxes shapes must remain identical. '
        'Do not use a whole-frame box merely to bypass reuse. Use save_detection() for unchanged masks, annotations or a '
        'supported background mask. Background-only changes need no contour revision, but record the measured background method. '
        'No supported baseline means relative heights/volumes remain unknown; never invent one.')
    reports=[]; last=None; last_error=''
    retained_result=copy_results([seed])[0] if seeded else None
    def retain_measured(message):
        verify(check_time=False)
        if retained_result is None:
            from ai_recovery import TotalTimeLimit
            raise TotalTimeLimit(message)
        retained=incomplete_measurements([retained_result], message, keep_flags=refinement is not None,
            audit=dict(detection_start_mode=seed_audit['detection_start_mode'],execution_mode='retained_at_total_limit',
                       provider=client.settings.get('provider','codex'),model=getattr(client,'model',client.settings.get('model'))))[0]
        retained['instructions']=instructions
        return retained
    budget.checkpoint=retain_measured
    measured_previous=seed if seeded else None
    checkpoint_inputs={}
    if refinement is not None:
        payload['task']=REFINEMENT_RULE+' '+seed_rule+' Write analysis.py and run it once to verify the changes.'
        payload['refinement']=refinement['request']
        payload['previous_result_file']='refinement.json'
        payload['helpers']+=' Start from load_checkpoint(0); save_checkpoint(0) restores the current masks unchanged. Use save() for justified repairs.'
        payload['requirements'][2]=seed_rule
        checkpoint_inputs=deepcopy(seed['audit']['checkpoint_inputs'])
        seed_audit.update(detection_start_mode=seed['audit'].get('detection_start_mode','default_detection'),refinement=True)
        if 'normal_detection' in seed_audit: seed_audit['normal_detection']['origin']='previous reviewed result (refinement input)'
    elif seeded:
        report(12,f'AI inspecting the image and suitability of {len(seed["particles"])} reference detections…')
        client.settings['timeout']=max(1,int(budget.remaining()))
        review=check_reply(timed('AI normal-detection review',lambda:client.run_agent(dict(task='Review the supplied normal particle detection for this single frame. '
            'Read normal_detection.json and compare input.png with normal_preview.png. '+seed_rule+' '
            'Accept if the existing detection meets the instructions; acceptance keeps its masks and flags exactly. '
            'Request revision only for concrete defects, a needed uncertainty annotation, or a supported background measurement '
            'requested by the user. Explain specific regions and changes. Do not generate code or modify files in this review.',
            numerical_metadata=meta,user_instructions=instructions,normal_detection_count=len(seed['particles']),
            missing_baseline=sum(p['background_nm'] is None for p in seed['particles'])),
            [str(root/'input.png'),str(root/'normal_preview.png')],result_schema(True),lambda msg:report(12,msg),writable=False)),True)
        reports.append(review);verify()
        if review['decision']=='accept':
            raw,changes=check_linked_detection(seed,seed['labels'],normal_raw)
            rows,_=timed('Local particle measurements',lambda:measure_reusing_previous(
                snapshot,seed['labels'],seed['background'],raw,seed,cancelled))
            warnings=list(dict.fromkeys(seed['warnings']+review['warnings']))
            if any(p['background_nm'] is None for p in rows):
                warnings.append('Normal detection retained. No supported local background: relative height/volume remain unavailable where indicated.')
            result=dict(snapshot=snapshot,labels=seed['labels'].copy(),background=seed['background'].copy(),particles=rows,
                selected_ids=selected_ids(seed),origin='ai',algorithm=normal_raw['algorithm'],parameters=normal_raw['parameters'],
                warnings=warnings,summary=review['summary'],ai_accepted=True,instructions=instructions,
                audit=dict(**seed_audit,detection_linking=changes,metadata=meta,raw_result=raw,reports=deepcopy(reports),
                    support_script=support,provider=client.settings.get('provider', 'codex'),model=getattr(client,'model',client.settings.get('model')),
                    reasoning_effort=getattr(client,'reasoning_effort',None),execution_mode='normal-detection-review',ai_request_count=1,
                    stage_timings=deepcopy(timings)))
            verify();report(100,f'AI accepted normal detection: {len(rows)} contours reused; no re-detection or generated script.')
            return result
        payload['task']='Choose and implement a suitable detection method for the problems identified by the review. '+seed_rule+' Write analysis.py and run it.'
        payload['normal_detection_review']=review
        payload['helpers']+=' load_detection() returns labels,background,normal metadata. save_detection(annotations,background,parameters,warnings) preserves contours. Use save() only for declared local contour repairs.'
        payload['requirements'][2]=seed_rule
        report(15,'AI requested a bounded repair of normal detection: '+review['summary'])
    payload.setdefault('requirements',[]).append(REPLAY_RULE)
    base_payload=deepcopy(payload)
    for attempt in range(max_rounds):
        check(); client.settings['timeout']=max(1,int(budget.remaining()))
        n=20+attempt*24 if seeded else 8+attempt*27
        report(n, (f'AI repairing normal detection' if seeded else 'AI inspecting this frame and testing segmentation')+f' — round {attempt+1}/{max_rounds}…')
        images=[str(root/'input.png')]+([str(root/'normal_preview.png')] if seeded else [])
        answer=check_reply(timed('AI local repair' if seeded else 'AI detection design',
            lambda:client.run_agent(payload,images,result_schema(),lambda msg:report(n,msg))))
        reports.append(answer); verify()
        try:
            script=read_workspace_file(root,'analysis.py',1024*1024)
            if not script.strip(): raise ValueError('Empty analysis.py')
            for name in ('result.json','labels.npy','background.npy'): (root/name).unlink(missing_ok=True)
            with isolated_replay(root,immutable,discard=('result.json','labels.npy','background.npy')):
                timed('Sandbox script verification',lambda:client.execute_analysis(command, timeout=replay_timeout(client,(1,*snapshot.image.shape))))
            verify()
            labels, background, raw=validate_segmentation(root,snapshot,input_hash)
            changes={}
            if seeded:
                raw,changes=check_linked_detection(seed,labels,raw,preserve_uncertainty=refinement is None)
                report(n,f'{changes["reused_particles"]} contours reused; {changes["revised_particles"]} revised locally.')
            records,reused=timed('Local particle measurements',lambda:measure_reusing_previous(
                snapshot,labels,background,raw,measured_previous,cancelled))
            measured_previous=dict(snapshot=snapshot,labels=labels,background=background,particles=records,
                audit=dict(raw_result=deepcopy(raw),checkpoint_inputs=deepcopy(checkpoint_inputs)))
            save_measured_checkpoint(root,attempt+1,[measured_previous],immutable)
            checkpoint_inputs=measured_previous['audit']['checkpoint_inputs']
            retained_result=dict(measured_previous,algorithm=raw['algorithm'],parameters=deepcopy(raw['parameters']),
                warnings=list(raw['warnings']),summary='Measured result awaiting AI review.',
                origin='ai',ai_accepted=False,instructions=instructions)
            retained_result['audit'].update(script=script,script_sha256=digest(root/'analysis.py'),support_script=support,
                metadata=meta,reports=deepcopy(reports),provider=client.settings.get('provider','codex'))
            if reused:report(n,'Contours and background unchanged: reusing geometry and height measurements.')
        except (ValueError, OSError, KeyError, CodexError) as exc:
            if isinstance(exc, TimeoutError) or is_timeout_message(exc): raise
            check(); last_error=str(exc)
            payload={**base_payload,'task':'Repair only the validation failure. Reuse the latest measured checkpoint '
                'when available, keeping earlier valid corrections. Regenerate outputs through the save helpers. '
                +(seed_rule if seeded else ''),'error':last_error,'latest_checkpoint':max(map(int,checkpoint_inputs),default=None)}
            continue
        report(n+13,'pyNuD measured particle geometry and heights. AI is reviewing the measured outlines…')
        with open_host_output(root,'measured.png') as stream: render(snapshot,records,long_edge=1200).save(stream,format='PNG')
        hashes={name:digest(root/name) for name in ('analysis.py','labels.npy','background.npy','result.json','measured.png')}
        client.settings['timeout']=max(1,int(budget.remaining()))
        client.progress=lambda msg:report(n+15,msg)
        review=review_measurements(lambda:timed('AI measured-result review',lambda:client.advise(dict(task='Review original vs HOST-measured contours for this single frame. '
            'Check missed weak particles, false positives, merged/split objects and array orientation. '
            'Do not compare with a desired count. Request revision when evidence warrants it. '
            'Human review is still required for uncertain and edge particles. Use only supplied images and metrics; '
            'do not execute code or repeat numerical analysis. '+(seed_rule if seeded else ''),
            detection_changes=changes,refinement_request=refinement['request'] if refinement else None,
            algorithm=raw['algorithm'],parameters=raw['parameters'],warnings=raw['warnings'],
            counts=dict(total=len(records),uncertain=sum(p['uncertain'] for p in records),edge=sum(p['edge'] for p in records)),
            missing_baseline=sum(p['background_nm'] is None for p in records),user_instructions=instructions),
            [str(root/'input.png'),str(root/'measured.png')],result_schema(True))),verify)
        reports.append(review); verify()
        if any((root/k).is_symlink() or digest(root/k)!=v for k,v in hashes.items()):
            raise ValueError('Outputs changed during read-only review; rejected.')
        last=dict(snapshot=snapshot,labels=labels,background=background,particles=records,algorithm=raw['algorithm'],
            parameters=raw['parameters'],warnings=list(dict.fromkeys(raw['warnings']+answer['warnings']+review['warnings'])),
            summary=review['summary'],ai_accepted=review['decision']=='accept',instructions=instructions,
            audit=dict(**seed_audit,metadata=meta,script=script,script_sha256=hashes['analysis.py'],raw_result=raw,reports=deepcopy(reports),
                       checkpoint_inputs=deepcopy(checkpoint_inputs),measurements_reused=reused,
                       support_script=support,provider=client.settings.get('provider', 'codex'),model=getattr(client,'model',client.settings.get('model')),
                       reasoning_effort=getattr(client,'reasoning_effort',None),ai_request_count=len(reports),stage_timings=deepcopy(timings)))
        if seeded:last['audit'].update(detection_linking=changes,execution_mode='refinement' if refinement is not None else 'normal-detection-local-repair')
        retained_result=last
        if review.get('review_incomplete'):
            last['audit']['ai_review_status']='incomplete'
            if refinement is None:
                for p in last['particles']:p['uncertain']=True
            break
        if review['decision']=='accept': break
        payload={**base_payload,'task':f'Repair only the concrete problems in this review, starting from load_checkpoint({attempt+1}) '
            'from analysis_support. It returns the latest verified labels, background and raw metadata. '
            'Copy the arrays before changing them. Keep correct contours and earlier corrections; revise only affected regions '
            'or annotations. Do not repeat whole-frame parameter searches. Execute once and inspect changed outlines. '
            +(seed_rule if seeded else ''),'review':review,'latest_checkpoint':attempt+1}
    else:
        if last is None: raise ValueError('No valid measured segmentation was produced: '+last_error)
        last['ai_accepted']=False
        last['warnings'].append('AI did not finish an accepted revision. All candidates require human review.')
        if refinement is None:
            for p in last['particles']: p['uncertain']=True
    check(); verify(); report(100,'Analysis complete. Review purple and blue candidates, then export the selected particles.')
    return last


def copy_results(results):
    """Deep copy masks and decisions while sharing the immutable captured images."""
    return [{**deepcopy({k: v for k, v in r.items() if k != 'snapshot'}), 'snapshot': r['snapshot']} for r in results]


def incomplete_measurements(results, message, *, keep_flags=False, audit=None):
    """An in-memory checkpoint, never the partially rewritten agent outputs.

    keep_flags leaves per-particle uncertainty as validated, so a Refine that
    cannot finish neither erases the localized triage nor the user's selections.
    Without it every candidate is uncertain and nothing stays selected for Use.
    """
    results = copy_results(results)
    for result in results:
        result['ai_accepted'] = False
        result['summary'] = message
        result.setdefault('warnings', []).append(message)
        result.setdefault('audit', {}).update(audit or {}, ai_review_status='timed_out')
        if not keep_flags:
            result.pop('selected_ids', None)
            for particle in result['particles']:
                particle['uncertain'] = True
    return results


def export_result(result, selected, target):
    """Explicit atomic export: CSV selected rows or JSON complete review snapshot."""
    target=Path(target); selected=set(selected)
    rows=result['particles']; ids={p['id'] for p in rows}
    if not selected <= ids: raise ValueError('Selection includes an unknown particle.')
    if target.suffix.lower() not in ('.csv','.json'): raise ValueError('Choose a CSV or JSON file.')
    if target.suffix.lower()=='.csv' and not selected: raise ValueError('Select at least one particle before exporting CSV.')
    source=result['snapshot']
    if source.metadata.get('source_path') and target.resolve()==Path(source.metadata['source_path']).resolve():
        raise ValueError('The source data cannot be overwritten.')
    source_unchanged(source)
    fd,name=tempfile.mkstemp(prefix='.pynud-particle-',suffix=target.suffix,dir=target.parent);os.close(fd)
    try:
        if target.suffix.lower()=='.json':
            data=frame_document(result,selected)
            Path(name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        else:
            fields=['source','frame_1based','channel']+[k for k in rows[0] if k not in ('polygons_xy','bbox_xyxy')]
            with open(name,'w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fields);writer.writeheader()
                for row in rows:
                    if row['id'] in selected:
                        writer.writerow(dict(source=Path(source.metadata.get('source_path','')).name,
                            frame_1based=source.metadata['frame_0based']+1,channel=source.metadata.get('channel',1),
                            **{k:v for k,v in row.items() if k in fields}))
        source_unchanged(source); os.replace(name,target)
    finally: Path(name).unlink(missing_ok=True)


def frame_document(result, selected, compact=False):
    from main_view_data import pack_frame
    from datetime import datetime, timezone
    from importlib.metadata import version, PackageNotFoundError
    import sys
    source=result['snapshot'];rows=result['particles']
    versions = {}
    for package in ('numpy', 'scipy', 'scikit-image', 'opencv-python', 'Pillow'):
        try: versions[package] = version(package)
        except PackageNotFoundError: pass
    data=dict(format=FORMAT,saved_at=datetime.now(timezone.utc).isoformat(),source=source.metadata,
        pixel_size_nm=list(source.pixel_size),shape=list(source.image.shape),display=source.display,
        palette=source.palette.tolist(),frame=pack_frame(source.image),
        labels=pack_frame(result['labels']) if compact else result['labels'].tolist(),
        background=pack_frame(result['background']) if compact else result['background'].tolist(),particles=rows,
        selected_ids=sorted(selected),algorithm=result['algorithm'],parameters=result['parameters'],
        instructions=result['instructions'],warnings=result['warnings'],summary=result['summary'],
        ai_accepted=result['ai_accepted'],audit=result['audit'],
        runtime=dict(python=sys.version,library_versions=versions))
    data['origin']=result.get('origin','ai')
    return data


SESSION_FORMAT = 'pynud-particle-analysis-session-v1'
MAX_JSON_BYTES = 512 * 1024 * 1024


def selected_ids(result):
    return set(result.get('selected_ids', [p['id'] for p in result['particles'] if not p['uncertain'] and not p['edge']]))


def apply_warnings(session):
    """Reasons a person should confirm before Apply; empty when nothing is flagged."""
    lines = []; statuses = set(); unaccepted = 0; uncertain = 0
    for r in session['frames']:
        if r.get('origin') == 'skipped': continue
        status = r.get('audit', {}).get('ai_review_status')
        if status in ('incomplete', 'timed_out', 'revision_requested', 'pending', 'validation_failed',
                      'review_failed', 'provider_error'):
            statuses.add(status)
        elif r.get('origin', 'ai') == 'ai' and not r.get('ai_accepted', True):
            unaccepted += 1
        chosen = selected_ids(r)
        uncertain += sum(bool(p['uncertain']) for p in r['particles'] if p['id'] in chosen)
    if statuses: lines.append('AI review status: ' + ', '.join(sorted(statuses)) + '.')
    if unaccepted: lines.append(f'{unaccepted} frame(s) were not accepted by the AI review.')
    if uncertain: lines.append(f'{uncertain} selected (Use) particle(s) are marked uncertain (purple).')
    return lines


def make_session(results, scope='current', id_scope='frame-local'):
    if not results: raise ValueError('No frame results to display.')
    frames = sorted(results, key=lambda r:r['snapshot'].metadata['frame_0based'])
    indices = [r['snapshot'].metadata['frame_0based'] for r in frames]
    if len(set(indices)) != len(indices): raise ValueError('Duplicate frame indices in analysis.')
    if scope not in ('current','all') or id_scope not in ('frame-local','movie-global'):
        raise ValueError('Unsupported analysis scope.')
    by_index=dict(zip(indices,frames))
    for result in frames:
        if result.get('origin')!='skipped':continue
        info=result.get('audit',{}).get('frame_sampling',{})
        index=result['snapshot'].metadata['frame_0based'];ref=info.get('reference_frame_0based')
        if (info.get('status')!='skipped' or info.get('frame_0based')!=index or type(ref)is not int
                or ref>=index or ref not in by_index or by_index[ref].get('origin')=='skipped'
                or not isinstance(info.get('overlap_fraction'),(int,float))
                or not FIELD_OVERLAP_MIN<=info['overlap_fraction']<=1
                or result['particles'] or np.any(result['labels']) or np.any(result['background'])
                or result['ai_accepted']):
            raise ValueError('Invalid skipped-frame record: skipped frames cannot contain analyzed particles.')
    session=dict(format=SESSION_FORMAT, scope=scope, id_scope=id_scope, frames=frames)
    if id_scope=='movie-global':
        first=first_observations(session)
        doubtful={p['id'] for r in frames for p in r['particles'] if p.get('identity_uncertain')}
        saved={}
        for r in frames:
            for p in r['particles']:
                p['first_frame_1based']=first[p['id']][0]['snapshot'].metadata['frame_0based']+1
                if 'selected_ids' in r:
                    state=p['id'] in selected_ids(r)
                    if p['id'] in saved and saved[p['id']]!=state:
                        raise ValueError('Inconsistent selection for a movie particle ID.')
                    saved[p['id']]=state
        # Identity ambiguity anywhere also flags the first observation used for statistics.
        for ident in doubtful:
            p=first[ident][1]
            if not p.get('identity_uncertain'):
                p['reason']+=' Identity correspondence requires review in later observations of this ID.'
            p['uncertain']=True;p['identity_uncertain']=True
        chosen={ident for ident,(r,p) in first.items() if saved.get(ident,not p['uncertain'] and not p['edge'])}
        for r in frames:r['selected_ids']=chosen & {p['id'] for p in r['particles']}
    return session


def first_observations(session):
    first={}
    for result in sorted(session['frames'],key=lambda r:r['snapshot'].metadata['frame_0based']):
        for particle in result['particles']:
            first.setdefault(particle['id'],(result,particle))
    return first


def set_session_selection(session, result, chosen):
    """Changing a Use checkbox changes that particle throughout a movie."""
    if session['id_scope']=='movie-global':
        ids={p['id'] for p in result['particles']}
        for frame in session['frames']:
            frame['selected_ids']=((selected_ids(frame)-ids)|set(chosen)) & {p['id'] for p in frame['particles']}
    else:result['selected_ids']=set(chosen)


def statistics_rows(session):
    """Never substitute a later measurement for an excluded first appearance."""
    observations=(first_observations(session).values() if session['id_scope']=='movie-global' else
                  ((r,p) for r in session['frames'] for p in r['particles']))
    return [(r,p) for r,p in observations if p['id'] in selected_ids(r)]


REFINEMENT_RULE = (
    'Revise the CURRENT Particle Analysis result according to the user correction instructions. '
    'Read refinement.json: it contains existing masks, measured particles, human Use choices, and previous code as '
    'reference text ONLY. Do not execute an old script automatically. Reuse the verified checkpoint 0; do not run '
    'the initial detector or rebuild supported ID links. Choose a different detection method only where the images '
    'and requested correction warrant it. Correct missing/merged/split particles, false positives, boundaries or '
    'ID correspondence as needed. The displayed frame and selected ID provide context, NOT a single-frame limit. '
    'Honor source frame numbers/ranges in the instructions, propagate necessary ID corrections, and preserve unrelated '
    'results. Human deselection is a review choice, not permission to delete a candidate. Justify removals and changes '
    'with image evidence; measure new geometry from the supplied heights. Do not force a requested count without evidence. '
    'The final read-only review must check whether the requested corrections were achieved.')


_REFINEMENT_FIELDS = ('labels','background','particles','selected_ids','algorithm','parameters','instructions',
                      'warnings','summary','ai_accepted','audit','origin')


def refinement_state(session):
    """Linear undo history: masks/decisions only, sharing the unchanged input images."""
    from main_view_data import pack_frame
    frames=[]
    for r in session['frames']:
        item={k:deepcopy(r[k]) for k in _REFINEMENT_FIELDS if k in r and k not in ('labels','background','selected_ids')}
        item.update(labels=pack_frame(r['labels']),background=pack_frame(r['background']),
            selected_ids=sorted(selected_ids(r)),frame_0based=r['snapshot'].metadata['frame_0based'],origin=r.get('origin','ai'))
        frames.append(item)
    return dict(scope=session['scope'],id_scope=session['id_scope'],frames=frames)


def prepare_refinement_inputs(root, refinement, seeds, immutable, movie=False):
    """Protect the previous result and replayable masks; never execute imported code."""
    (root/'refinement.json').write_text(json.dumps(refinement,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    immutable['refinement.json']=digest(root/'refinement.json')
    for seed in seeds:
        seed['audit']['raw_result']=dict(algorithm=seed['algorithm'],
            parameters={**deepcopy(seed['parameters']),'detection_revisions':[]},warnings=list(seed['warnings']),
            particles=[dict(id=p['id'],uncertain=p['uncertain'],identity_uncertain=p.get('identity_uncertain',False),
                near_edge=p.get('near_edge',False),reason=p['reason']) for p in seed['particles']])
    save_measured_checkpoint(root,0,seeds,immutable,movie=movie)


def refine_session(session, client, workspace, instructions, progress=lambda p,m:None,
                   cancelled=lambda:False, time_limit=900, *, focus_frame=None, focus_id=None, max_rounds=2):
    """Refine a captured review without redetection, recapture, or implicit Apply."""
    instructions=instructions.strip()
    if not instructions:raise ValueError('Describe what the AI should correct before refining.')
    check_cancel(cancelled)
    frames=session['frames'];seeds=[]
    for r in frames:
        source_unchanged(r['snapshot'])
        if r.get('origin')=='skipped':continue
        validate_masks(r['labels'],r['background'],r['particles'])
        seeds.append({**deepcopy({k:v for k,v in r.items() if k!='snapshot'}),'snapshot':r['snapshot']})
    if not seeds:raise ValueError('No analyzed frames to refine.')
    if focus_frame is None:focus_frame=seeds[0]['snapshot'].metadata['frame_0based']+1
    focus=next((r for r in seeds if r['snapshot'].metadata['frame_0based']+1==focus_frame),None)
    if focus is None:raise ValueError('Select an analyzed frame to refine. Skipped frames remain skipped.')
    if focus_id is not None and focus_id not in {p['id'] for p in focus['particles']}:
        raise ValueError('The selected particle no longer exists.')
    before=refinement_state(session)
    request=dict(instructions=instructions,frame_1based=focus_frame,selected_id=focus_id,id_scope=session['id_scope'],
        analyzed_frames_1based=[r['snapshot'].metadata['frame_0based']+1 for r in seeds],
        skipped_frames_1based=[r['snapshot'].metadata['frame_0based']+1 for r in frames if r.get('origin')=='skipped'])
    refinement=dict(request=request,previous_result=before)
    progress(0,'Reusing current masks, measurements and particle IDs for AI refinement…')
    if session['scope']=='current' and len(seeds)==1:
        result=run_analysis(seeds[0]['snapshot'],client,workspace,instructions,progress,cancelled,
            time_limit=time_limit,max_rounds=max_rounds,initial_detection=seeds[0],refinement=refinement)
        revised=make_session([result],session['scope'],session['id_scope'])
    else:
        plan=[dict(frame_0based=r['snapshot'].metadata['frame_0based'],status='analyzed' if
            r.get('origin')!='skipped' else 'skipped') for r in frames] if has_skipped_frames(session) else None
        revised=run_movie_analysis([r['snapshot'] for r in seeds],client,workspace,instructions,progress,cancelled,
            time_limit=time_limit,max_rounds=max_rounds,initial_detections=seeds,frame_sampling=plan,refinement=refinement)
        revised['frames'].extend({**deepcopy({k:v for k,v in r.items() if k!='snapshot'}),'snapshot':r['snapshot']}
            for r in frames if r.get('origin')=='skipped')
    # Preserve user choices only for IDs whose geometry, background and flags are
    # unchanged in every observation. Revised/new identities start unchecked.
    old={r['snapshot'].metadata['frame_0based']:r for r in frames};changed=set()
    for r in revised['frames']:
        prior=old[r['snapshot'].metadata['frame_0based']]
        a={p['id']:p for p in prior['particles']};b={p['id']:p for p in r['particles']}
        changed_here=set(a)^set(b)
        for ident in set(a)&set(b):
            if (session['id_scope']!=revised['id_scope'] or not np.array_equal(prior['labels']==ident,r['labels']==ident)
                    or not np.array_equal(prior['background'],r['background'])
                    or any(a[ident].get(k)!=b[ident].get(k) for k in ('uncertain','edge','identity_uncertain'))):
                changed_here.add(ident)
        r['selected_ids']=selected_ids(prior)&set(b)-changed_here
        r['audit']['refinement_request']=deepcopy(request)
        r['audit']['refinement_requires_review']=sorted(changed_here&set(b))
        changed.update(changed_here)
    if revised['id_scope']=='movie-global':
        for r in revised['frames']:
            r['selected_ids']-=changed
            r['audit']['refinement_requires_review']=sorted(changed&{p['id'] for p in r['particles']})
    revised=make_session(revised['frames'],revised['scope'],revised['id_scope'])
    revised['refinement_history']=deepcopy(session.get('refinement_history',[]))+[before]
    check_cancel(cancelled)
    for r in frames:source_unchanged(r['snapshot'])
    return revised


def undo_refinement(session, cancelled=lambda:False):
    history=session.get('refinement_history',[])
    if not history:raise ValueError('There is no previous refinement to restore.')
    previous=history[-1]
    current={r['snapshot'].metadata['frame_0based']:r for r in session['frames']}
    items=previous.get('frames',[])
    if len(items)!=len(current) or {i.get('frame_0based') for i in items}!=set(current):
        raise ValueError('Invalid refinement history frames.')
    results=[]
    for item in items:
        check_cancel(cancelled)
        r=current[item['frame_0based']]
        doc=frame_document(r,selected_ids(r),compact=True)
        doc.update({k:deepcopy(item[k]) for k in _REFINEMENT_FIELDS if k in item})
        restored=restore_frame(doc,cancelled);restored['snapshot']=r['snapshot'];results.append(restored)
    restored=make_session(results,previous['scope'],previous['id_scope'])
    if len(history)>1:restored['refinement_history']=deepcopy(history[:-1])
    return restored


def export_session(session, target):
    """Portable input/mask/ID records. Never execute code embedded in a JSON."""
    target=Path(target)
    if target.suffix.lower() not in ('.json','.csv'): raise ValueError('Choose JSON or CSV.')
    frames=session['frames']
    if not frames: raise ValueError('No frame results to export.')
    for result in frames:
        source=result['snapshot'];source_unchanged(source)
        if source.metadata.get('source_path') and target.resolve()==Path(source.metadata['source_path']).resolve():
            raise ValueError('The source data cannot be overwritten.')
        if not selected_ids(result) <= {p['id'] for p in result['particles']}:raise ValueError('Unknown selected ID.')
    fd,tmp=tempfile.mkstemp(prefix='.pynud-particles-',suffix=target.suffix,dir=target.parent);os.close(fd)
    try:
        if target.suffix.lower()=='.json':
            # Stream frame records; do not build a second full-movie JSON object.
            with open(tmp,'w',encoding='utf-8') as stream:
                head=dict(format=SESSION_FORMAT,scope=session['scope'],id_scope=session['id_scope'],
                          statistics_policy=('first-analyzed-appearance' if has_skipped_frames(session) else 'first-appearance')
                              if session['id_scope']=='movie-global' else 'each-observation')
                if session.get('refinement_history'):head['refinement_history']=session['refinement_history']
                stream.write(json.dumps(head)[:-1]+', "frames": [')
                for index,result in enumerate(frames):
                    if index:stream.write(',')
                    json.dump(frame_document(result,selected_ids(result),compact=True),stream,ensure_ascii=False,allow_nan=False)
                stream.write(']}')
        else:
            fields=['source','frame_1based','id_scope','statistics_policy','id','first_frame_1based','identity_uncertain','uncertain','edge','reason','area_nm2','equivalent_diameter_nm',
                'perimeter_nm','circularity','major_axis_nm','minor_axis_nm','aspect_ratio','orientation_deg',
                'min_z_nm','max_z_nm','mean_z_nm','std_z_nm','background_nm','background_pixels','max_above_bg_nm',
                'mean_above_bg_nm','volume_above_bg_nm3','x_nm','y_nm']
            with open(tmp,'w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fields);writer.writeheader()
                for result,p in statistics_rows(session):
                    meta=result['snapshot'].metadata
                    row={k:p.get(k) for k in fields};row.update(source=Path(meta.get('source_path','')).name,
                        frame_1based=meta['frame_0based']+1,id_scope=session['id_scope'],
                        statistics_policy=('first-analyzed-appearance' if has_skipped_frames(session) else 'first-appearance')
                            if session['id_scope']=='movie-global' else 'each-observation')
                    writer.writerow(row)
        for result in frames:source_unchanged(result['snapshot'])
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)


def restore_frame(data, cancelled=lambda:False):
    """Validate portable pixels/masks; rebuild contours and measurements locally."""
    from main_view_data import unpack_frame
    if not isinstance(data,dict) or data.get('format')!=FORMAT:raise ValueError('Unsupported particle-frame format.')
    shape=data.get('shape')
    if (not isinstance(shape,list) or len(shape)!=2 or any(type(n) is not int or n<4 for n in shape)
            or math.prod(shape)>MAX_PIXELS):raise ValueError('Invalid frame shape.')
    meta=data.get('source',{})
    if (not isinstance(meta,dict) or type(meta.get('frame_0based')) is not int or not 0<=meta['frame_0based']<10000000
            or meta.get('channel',1)!=1 or not isinstance(meta.get('source_path',''),str)):
        raise ValueError('Invalid frame metadata.')
    palette=np.asarray(data.get('palette'))
    if palette.shape!=(256,3) or palette.dtype.kind not in 'iu' or palette.min()<0 or palette.max()>255:
        raise ValueError('Invalid color palette.')
    check_cancel(cancelled)
    image=unpack_frame(data['frame'],shape)
    labels=(unpack_frame(data['labels'],shape) if isinstance(data['labels'],dict) else np.asarray(data['labels']))
    bg=(unpack_frame(data['background'],shape) if isinstance(data['background'],dict) else np.asarray(data['background']))
    if (labels.shape!=tuple(shape) or labels.dtype.kind not in 'iuf' or not np.isfinite(labels).all()
            or np.any(labels!=np.floor(labels)) or labels.min()<0 or labels.max()>MAX_PARTICLES):
        raise ValueError('Invalid segmentation labels.')
    if bg.shape!=tuple(shape) or bg.dtype.kind not in 'biuf' or not np.isin(bg,[0,1]).all():
        raise ValueError('Invalid background mask.')
    labels=labels.astype(np.int32);bg=bg.astype(bool)
    raw_particles=data.get('particles')
    if not isinstance(raw_particles,list) or len(raw_particles)>MAX_PARTICLES:raise ValueError('Invalid particle records.')
    rings=[p for p in raw_particles if isinstance(p,dict) and p.get('geometry_kind')=='ring_contours']
    segmented=[p for p in raw_particles if p not in rings]
    validate_masks(labels,bg,segmented)
    display=data.get('display')
    if not isinstance(display,dict):raise ValueError('Invalid display settings.')
    meta=deepcopy(meta);meta['detached_snapshot']=True
    snap=FrameInput(image,tuple(data['pixel_size_nm']),palette,display,meta)
    # Validate contrast/LUT even when a file contains no detections.
    from main_view_data import display_tone_pixels
    display_tone_pixels(snap.image,snap.display)
    records=measure_particles(snap,labels,bg,dict(particles=segmented),cancelled)
    if rings:
        if len(rings)!=1 or records:raise ValueError('Invalid ring-only frame.')
        records=[ring_record(snap,rings[0]['polygons_xy'],rings[0]['id'])]
    ids={p['id'] for p in records};chosen=data.get('selected_ids')
    if not isinstance(chosen,list) or any(type(i)is not int for i in chosen) or len(set(chosen))!=len(chosen) or not set(chosen)<=ids:
        raise ValueError('Invalid selected particle IDs.')
    for key in ('algorithm','summary','instructions'):
        if not isinstance(data.get(key),str):raise ValueError('Invalid analysis description.')
    if not isinstance(data.get('parameters'),dict) or not isinstance(data.get('audit'),dict):raise ValueError('Invalid analysis record.')
    if not isinstance(data.get('warnings'),list) or any(not isinstance(s,str) for s in data['warnings']):raise ValueError('Invalid warnings.')
    if type(data.get('ai_accepted')) is not bool:raise ValueError('Invalid review state.')
    result={key:deepcopy(data[key]) for key in ('algorithm','summary','instructions','parameters','audit','warnings','ai_accepted')}
    result.update(snapshot=snap,labels=labels,background=bg,particles=records,selected_ids=set(chosen),
                  origin=str(data.get('origin','ai')),imported=True)
    return result


def import_session(path, progress=lambda p,m:None, cancelled=lambda:False):
    path=Path(path)
    if path.stat().st_size>MAX_JSON_BYTES:raise ValueError('Analysis JSON is too large (maximum 512 MiB).')
    check_cancel(cancelled)
    data=json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Non-finite JSON value.')))
    if not isinstance(data,dict):raise ValueError('Invalid analysis JSON.')
    if data.get('format')==FORMAT:items=[data];scope='current';id_scope='frame-local'
    elif data.get('format')==SESSION_FORMAT:
        items=data.get('frames');scope=data.get('scope','all')
        id_scope=data.get('id_scope')
        if id_scope not in ('frame-local','movie-global'):raise ValueError('Unsupported ID scope.')
    else:raise ValueError('Choose a Particle Analysis JSON containing images, segmentation masks and IDs.')
    if not isinstance(items,list) or not 1<=len(items)<=10000:raise ValueError('Invalid frame collection.')
    total=0
    for item in items:
        shape=item.get('shape') if isinstance(item,dict) else None
        if not isinstance(shape,list) or len(shape)!=2 or any(type(n)is not int or n<4 for n in shape):raise ValueError('Invalid frame shape.')
        total+=math.prod(shape)
    if total>MAX_SESSION_PIXELS:raise ValueError('Analysis contains too many image pixels.')
    results=[]
    for i,item in enumerate(items):
        check_cancel(cancelled);progress(round(100*i/len(items)),f'Restoring frame {i+1}/{len(items)} locally…')
        results.append(restore_frame(item,cancelled))
    session=make_session(results,scope,id_scope)
    history=data.get('refinement_history',[])
    if not isinstance(history,list) or any(not isinstance(item,dict) for item in history):
        raise ValueError('Invalid refinement history.')
    if history:session['refinement_history']=history
    paths={r['snapshot'].metadata.get('source_path','') for r in results}
    if len(paths)>1:raise ValueError('An analysis session must refer to one source file.')
    session['import_path']=str(path.resolve())
    progress(100,'Imported segmentation and IDs. No AI request was made.')
    return session


def ring_record(snapshot, polygons, ident=1):
    """Persist measured ring curves without inventing a filled particle region."""
    if type(ident)is not int or not 1<=ident<=MAX_PARTICLES:raise ValueError('Invalid ring ID.')
    h,w=snapshot.image.shape;curves=[]
    for curve in polygons:
        a=np.asarray(curve,dtype=float)
        if (a.ndim!=2 or a.shape[1]!=2 or not 3<=len(a)<=100000 or not np.isfinite(a).all()
                or a[:,0].min()<-.5 or a[:,1].min()<-.5 or a[:,0].max()>w-.5 or a[:,1].max()>h-.5):
            raise ValueError('Invalid ring contours.')
        curves.append(a)
    if not 1<=len(curves)<=3:raise ValueError('No ring contours to save.')
    points=np.concatenate(curves);x,y=points.mean(axis=0);dx,dy=snapshot.pixel_size
    return dict(id=ident,uncertain=False,edge=bool(np.any(points[:,0]<=0) or np.any(points[:,1]<=0)
        or np.any(points[:,0]>=w-1) or np.any(points[:,1]>=h-1)),geometry_kind='ring_contours',
        reason='Manual ring contours only; no filled segmentation. Area, height and volume are not inferred from these curves.',
        x_px=float(x),y_px=float(y),x_nm=float((x+.5)*dx),y_nm=float((y+.5)*dy),
        background_nm=None,background_pixels=0,polygons_xy=[a.tolist() for a in curves])


def adaptive_watershed_detection(data, prominence=0.016, contour_fraction=0.35,
                                 size_scale=1.0, valid_mask=None, normalize=True,
                                 reference_shape=None):
    """Local detector from the 2026-09-30 image trial; never modifies heights.

    Pixel lengths follow the trial's 600-pixel field. For an ROI, pass the full
    field shape so cropping does not change the assumed particle size. Mask and
    normalize=False are useful for reproducing the original screenshot trial.
    Labels retain every candidate; selection is a separate human-review step.
    """
    from scipy import ndimage
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    image = np.asarray(data, dtype=float)
    if image.ndim != 2 or min(image.shape) < 4:
        raise ValueError('Adaptive Watershed requires a 2-D image of at least 4 x 4 pixels.')
    if not (0 < prominence <= 1 and 0 < contour_fraction < 1 and 0.1 <= size_scale <= 10):
        raise ValueError('Invalid Adaptive Watershed parameters.')
    valid = np.isfinite(image)
    if valid_mask is not None:
        if np.shape(valid_mask) != image.shape:
            raise ValueError('Detection mask and image shapes differ.')
        valid &= np.asarray(valid_mask, dtype=bool)
    labels = np.zeros(image.shape, dtype=np.int32)
    scale = max(0.125, min(reference_shape or image.shape) / 600.0) * size_scale
    parameters = dict(algorithm='adaptive_watershed_v1', prominence=prominence,
                      contour_fraction=contour_fraction, size_scale=size_scale,
                      pixel_scale=scale, normalization='finite min-max' if normalize else 'none',
                      baseline_percentile=15, seed_sigma_px=2*scale,
                      stability_prominence=0.04, log_strength_threshold=0.035)
    if not valid.any() or np.ptp(image[valid]) <= 0:
        return labels, [], parameters
    # Nearest finite pixels avoid artificial depressions around missing data.
    if not np.isfinite(image).all():
        nearest = ndimage.distance_transform_edt(~np.isfinite(image), return_distances=False, return_indices=True)
        image = image[tuple(nearest)]
    else:
        image = image.copy()
    if normalize:
        lo, hi = image[valid].min(), image[valid].max()
        image = (image-lo)/(hi-lo)

    def peaks(values, height):
        regions, _ = ndimage.label(h_maxima(values, height) & valid)
        points = []
        for region in measure.regionprops(regions, intensity_image=values):
            coords = region.coords
            y, x = coords[np.argmax(values[coords[:, 0], coords[:, 1]])]
            if values[y, x] > 0.08:
                points.append((int(y), int(x)))
        return sorted(points)

    smooth = ndimage.gaussian_filter(image, 2*scale)
    points = peaks(smooth, prominence)
    if not points:
        return labels, [], parameters
    if len(points) > 10000:
        raise ValueError('Too many particle candidates. Increase Peak prominence or Particle scale.')
    ys, xs = np.asarray(points).T
    markers = labels.copy()
    markers[ys, xs] = np.arange(1, len(points)+1)
    basins = watershed(-smooth, markers, mask=valid, connectivity=2)
    # Only retain scale-space responses at the seeds, not 23 full-size arrays.
    responses = np.array([(-ndimage.gaussian_laplace(image, s*scale)*(s*scale)**2)[ys, xs]
                          for s in range(3, 26)])
    stability = [peaks(ndimage.gaussian_filter(image, s*scale), 0.04) for s in (1.5, 2., 3.)]
    records = []
    h, w = image.shape
    for ident, bounds in enumerate(ndimage.find_objects(basins), 1):
        if bounds is None:
            continue
        y, x = points[ident-1]
        local_y, local_x = y-bounds[0].start, x-bounds[1].start
        basin = basins[bounds] == ident
        values = smooth[bounds]
        baseline = np.percentile(values[basin], 15)
        level = baseline+contour_fraction*(smooth[y, x]-baseline)
        components, _ = ndimage.label((values >= level) & basin)
        region = components == components[local_y, local_x]
        region = ndimage.binary_fill_holes(region) & basin
        labels[bounds][region] = ident
        ry, rx = np.nonzero(region)
        ry = ry+bounds[0].start; rx = rx+bounds[1].start
        diameter = np.sqrt(4*region.sum()/np.pi)
        tolerance = max(6*scale, min(12*scale, 0.2*diameter))
        support = sum(any((py-y)**2+(px-x)**2 <= tolerance**2 for py, px in ps) for ps in stability)
        best = int(np.argmax(responses[:, ident-1]))
        strength = float(responses[best, ident-1])
        edge = bool(np.any((rx <= 2*scale) | (rx >= w-1-2*scale) |
                          (ry <= 2*scale) | (ry >= h-1-2*scale)) or
                    y < 6*scale or y > h-1-6*scale or x < 6*scale or x > w-1-6*scale)
        weak = bool(support < 3 or strength < 0.035)
        records.append(dict(id=ident, uncertain=weak, near_edge=edge,
                            reason=('Near image edge; review partial coverage. ' if edge else '') +
                                   f'Peak support {support}/3; LoG strength {strength:.4f}.',
                            seed_xy=[x, y], stability_support=support, log_strength=strength,
                            log_sigma_px=(best+3)*scale, contour_level_normalized=float(level)))
    return labels, records, parameters

def manual_result(snapshot, labels, parameters, legacy_properties=(), ring_info=None,
                  candidate_records=None, selected_ids=None):
    """Use existing manual segmentation IDs and keep its original measurements."""
    labels=np.asarray(labels)
    if labels.shape!=snapshot.image.shape or labels.dtype.kind not in 'biu':raise ValueError('Manual labels do not match the analyzed image.')
    if labels.dtype.kind=='b':labels=measure.label(labels)
    labels=np.array(labels,dtype=np.int32,copy=True);bg=np.zeros(labels.shape,bool)
    records=(deepcopy(candidate_records) if candidate_records is not None else
        [dict(id=int(i),uncertain=False,reason='Manually configured detection; review the measured segmentation.') for i in np.unique(labels) if i])
    validate_masks(labels,bg,records)
    particles=measure_particles(snapshot,labels,bg,dict(particles=records))
    if not particles and ring_info:
        curves=[np.asarray(ring_info[key])[:,::-1] for key in ('mid_contour','inner_contour','outer_contour') if ring_info.get(key) is not None]
        if curves:particles=[ring_record(snapshot,curves)]
    def safe(value):
        if isinstance(value,np.ndarray):return safe(value.tolist())
        if isinstance(value,np.generic):return safe(value.item())
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [safe(v) for v in value]
        return value
    chosen=({p['id'] for p in particles} if selected_ids is None else set(selected_ids))
    if not chosen <= {p['id'] for p in particles}:raise ValueError('Unknown selected particle ID.')
    return dict(snapshot=snapshot,labels=labels,background=bg,particles=particles,selected_ids=chosen,
        origin='manual',algorithm='Manual Particle Analysis: '+str(parameters.get('method','')),
        parameters=safe(parameters),warnings=[],summary='Restorable segmentation from manual analysis. IDs are local to this frame. '
        'Original manual measurements are retained in the JSON audit record. Local background was not classified; relative height/volume are unavailable.',
        instructions='',ai_accepted=False,audit=dict(legacy_properties=safe(legacy_properties),ring_info=safe(ring_info),
                                                   candidate_records=safe(records)))


NORMAL_DETECTION_DEFAULTS = dict(method='Adaptive Watershed', prominence=.016, contour_fraction=.35,
    size_scale=1., background='None', rolling_radius_nm=50., smoothing='None', smoothing_size=1.)


def normal_detection_recipe(settings=None):
    recipe=dict(NORMAL_DETECTION_DEFAULTS)
    recipe.update({k:v for k,v in (settings or {}).items() if k in recipe})
    if (recipe['method']!='Adaptive Watershed' or recipe['background'] not in ('None','Rolling Ball','Polynomial Fit')
            or recipe['smoothing'] not in ('None','Gaussian','Median')):
        raise ValueError('Unsupported normal detection settings.')
    for key in ('prominence','contour_fraction','size_scale','rolling_radius_nm','smoothing_size'):
        recipe[key]=float(recipe[key])
        if not math.isfinite(recipe[key]) or recipe[key]<=0:raise ValueError('Invalid normal detection setting: '+key)
    return recipe


def preprocess_normal_detection(image, pixel_size, settings):
    """Same optional plugin preprocessing as Detect; measurement input stays intact."""
    settings=normal_detection_recipe(settings);data=np.array(image,dtype=float,copy=True)
    if settings['background']=='Rolling Ball':
        from skimage.restoration import rolling_ball
        background=rolling_ball(data,radius=max(1,int(settings['rolling_radius_nm']/min(pixel_size))))
        data=data-background
    elif settings['background']=='Polynomial Fit':
        from scipy.optimize import curve_fit
        yy,xx=np.indices(data.shape)
        def plane(xy,a,b,c):return a+b*xy[0]+c*xy[1]
        try:
            fit,_=curve_fit(plane,(xx.ravel(),yy.ravel()),data.ravel())
            data=data-plane((xx,yy),*fit)
        except (ValueError,RuntimeError):
            data=ndi.gaussian_filter(data,10);data-=ndi.gaussian_filter(data,10)
    if settings['smoothing']=='Gaussian':data=ndi.gaussian_filter(data,settings['smoothing_size'])
    elif settings['smoothing']=='Median':data=ndi.median_filter(data,max(1,int(settings['smoothing_size'])))
    return data


FIELD_OVERLAP_MIN = 0.90


def _overlap_registration(reference, current, minimum_overlap=FIELD_OVERLAP_MIN, minimum_ncc=.90):
    """Estimate a translation, then verify it on the actual shared pixels."""
    from skimage.registration import phase_cross_correlation
    if reference.shape != current.shape or min(current.shape) < 16:
        return None, dict(reason='Image dimensions cannot support overlap registration.')
    h, w = current.shape
    difference=current-reference
    if np.ptp(difference) <= max(1.,float(np.ptp(reference)))*1e-12:
        return (np.zeros(2),np.ones(current.shape,bool)),dict(overlap_fraction=1.,
            translation_yx_px=[0.,0.],registration_ncc=1.,registration_method='Identical data up to constant height offset')
    sigma = max(.6, min(h, w)/600.)
    def texture(image):
        smooth = ndi.gaussian_filter(image, sigma)
        return smooth-ndi.gaussian_filter(smooth, max(4., min(h, w)/20.))
    a, b = texture(reference), texture(current)
    if min(float(a.std()), float(b.std())) < 1e-10:
        return None, dict(reason='Insufficient texture for reliable registration.')
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)
    movement = -np.asarray(shift, dtype=float)  # reference -> current, in array y/x
    overlap = max(0., 1-abs(movement[0])/h)*max(0., 1-abs(movement[1])/w)
    info = dict(overlap_fraction=float(overlap), translation_yx_px=movement.tolist())
    if overlap < minimum_overlap:
        return None, dict(info, reason=f'Shared field is below {minimum_overlap:.0%}.')
    valid = ndi.shift(np.ones(a.shape, np.uint8), movement, order=0, mode='constant', cval=0, prefilter=False).astype(bool)
    warped = ndi.shift(a, movement, order=1, mode='constant', cval=0, prefilter=False)
    # Registration alone is not evidence that the particles stayed unchanged.
    av, bv = warped[valid], b[valid]
    av, bv = av-av.mean(), bv-bv.mean()
    ncc = float(np.dot(av, bv)/max(np.linalg.norm(av)*np.linalg.norm(bv), 1e-30))
    info['registration_ncc'] = float(np.clip(ncc,-1,1))
    if ncc < minimum_ncc:
        return None, dict(info, reason='Shared-region texture does not support the proposed translation.')
    # Repeated lattice patterns can have equally plausible translation peaks.
    from scipy.signal import fftconvolve
    correlation = fftconvolve(a, b[::-1, ::-1], mode='full')
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    radius = max(3, int(round(3*sigma)))
    yy, xx = np.ogrid[:correlation.shape[0], :correlation.shape[1]]
    outside = (yy-peak[0])**2+(xx-peak[1])**2 > radius**2
    ratio = float(correlation[peak]/max(float(correlation[outside].max()), 1e-30))
    info['registration_peak_ratio'] = ratio
    if ratio < 1.03:
        return None, dict(info, reason='Translation is ambiguous in repeated image structure.')
    return (movement, valid), info


def _repair_boxes(mask):
    """Cover changed pixels without turning an L-shaped new field into one full crop."""
    def split(local, y0, x0, depth=0):
        ys, xs=np.nonzero(local)
        if not len(ys):return []
        top,bottom,left,right=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        y0+=top;x0+=left;local=local[top:bottom,left:right]
        h,w=local.shape;best=(h*w,None,None)
        if depth<4 and h*w>64:
            for axis,view in enumerate((local,local.T)):
                n,m=view.shape
                occupied=view.any(axis=1)
                lo=np.where(occupied,view.argmax(axis=1),m)
                hi=np.where(occupied,m-view[:,::-1].argmax(axis=1),0)
                pre_lo=np.minimum.accumulate(lo);pre_hi=np.maximum.accumulate(hi)
                post_lo=np.minimum.accumulate(lo[::-1])[::-1];post_hi=np.maximum.accumulate(hi[::-1])[::-1]
                cuts=np.arange(1,n)
                costs=cuts*np.maximum(0,pre_hi[:-1]-pre_lo[:-1])+(n-cuts)*np.maximum(0,post_hi[1:]-post_lo[1:])
                if len(costs):
                    i=int(costs.argmin())
                    if costs[i]<best[0]:best=(int(costs[i]),axis,i+1)
        if best[0]<h*w*.7:
            _,axis,k=best
            if axis==0:return split(local[:k],y0,x0,depth+1)+split(local[k:],y0+k,x0,depth+1)
            return split(local[:,:k],y0,x0,depth+1)+split(local[:,k:],y0,x0+k,depth+1)
        return [[int(y0),int(x0),int(y0+h),int(x0+w)]]
    return split(np.asarray(mask,bool),0,0)


def select_movie_frames(snapshots, progress=lambda p,m:None, cancelled=lambda:False):
    """Cheap field screening before any detection/measurement or AI submission.

    Compare with the last retained frame, never a chain of skipped frames. This
    limits accumulated drift. Skipping is sampling, not proof of no new particles.
    """
    plan=[];reference=None
    for snap in snapshots:
        check_cancel(cancelled);source_unchanged(snap)
        index=snap.metadata['frame_0based']
        info=dict(frame_0based=index, status='analyzed', threshold=FIELD_OVERLAP_MIN,
                  reason='First representative frame.')
        if reference is not None:
            same=(snap.image.shape==reference.image.shape and snap.pixel_size==reference.pixel_size and
                  all(snap.metadata.get(k)==reference.metadata.get(k) for k in
                      ('source_path','source_size','source_mtime_ns','channel')))
            if same:
                registration,evidence=_overlap_registration(reference.image,snap.image)
                info.update(evidence,reference_frame_0based=reference.metadata['frame_0based'])
                if registration is not None:
                    info.update(status='skipped',reason='At least 90% overlap with a verified representative; detailed analysis skipped.')
            else:info['reason']='Source, dimensions or calibration changed; analyze this frame.'
        if info['status']=='analyzed':reference=snap
        plan.append(info)
        detail=(f'{info["overlap_fraction"]:.1%} overlap; ' if 'overlap_fraction' in info else '')
        progress(0,f'Frame {index+1}: {info["status"]} — '+detail+info['reason'])
    kept=sum(p['status']=='analyzed' for p in plan)
    progress(0,f'Overlap screening: {kept}/{len(plan)} frames to analyze; {len(plan)-kept} skipped. Skipped images are not sent to AI.')
    return plan


def complete_sampled_session(snapshots, session, plan, cancelled=lambda:False):
    """Retain skipped inputs for review, without inventing masks or measurements."""
    if [s.metadata['frame_0based'] for s in snapshots]!=[p['frame_0based'] for p in plan]:
        raise ValueError('Frame screening does not match the input movie.')
    analyzed={r['snapshot'].metadata['frame_0based']:r for r in session['frames']}
    if set(analyzed)!={p['frame_0based'] for p in plan if p['status']=='analyzed'}:
        raise ValueError('AI results do not cover the representative frames.')
    results=[]
    for snap,info in zip(snapshots,plan):
        check_cancel(cancelled);source_unchanged(snap)
        if info['status']=='analyzed':
            result=analyzed[info['frame_0based']]
            result['audit']['frame_sampling']=deepcopy(info)
        else:
            result=dict(snapshot=snap,labels=np.zeros(snap.image.shape,np.int32),
                background=np.zeros(snap.image.shape,bool),particles=[],selected_ids=set(),origin='skipped',
                algorithm='Skipped overlapping frame',parameters={},instructions='',ai_accepted=False,
                summary=f'Detailed analysis skipped: {info["overlap_fraction"]:.1%} overlap with frame {info["reference_frame_0based"]+1}.',
                warnings=['No detections, IDs or measurements were generated for this frame. Small changes and newly visible edge particles may be omitted.'],
                audit=dict(frame_sampling=deepcopy(info)))
        results.append(result)
    return make_session(results,'all','movie-global')


def has_skipped_frames(session):
    return any(r.get('origin')=='skipped' for r in session['frames'])


def _overlap_detection(reference, image, snapshot, recipe, cancelled=lambda:False):
    """Reuse unchanged masks from a fully detected anchor; repair changed areas.

    Every frame is checked. Warping always starts at the full-detection anchor,
    never a chain of rounded/interpolated masks. New field strips and changed
    topography require local detection. Broad changes fall back to full detection.
    """
    from main_view_data import pack_frame
    before, ref_image = reference
    base = before['snapshot']
    info = dict(reference_frame_0based=base.metadata['frame_0based'], minimum_overlap=FIELD_OVERLAP_MIN)
    if (base.pixel_size != snapshot.pixel_size or
            any(base.metadata.get(k) != snapshot.metadata.get(k) for k in
                ('source_path', 'source_size', 'source_mtime_ns', 'channel'))):
        return None, dict(info, reason='Source or calibration changed.')
    registration, registration_info = _overlap_registration(ref_image, image)
    info.update(registration_info)
    if registration is None:
        return None, info
    check_cancel(cancelled)
    movement, valid = registration
    h, w = image.shape
    scale = max(.125, min(h, w)/600.)*recipe['size_scale']
    # Compare smoothed numerical heights, allowing a constant height-origin shift.
    sigma = max(.6, 2*scale)
    smooth = ndi.gaussian_filter(image, sigma)
    ref_smooth = ndi.gaussian_filter(ref_image, sigma)
    aligned = ndi.shift(ref_smooth, movement, order=1, mode='constant', cval=0, prefilter=False)
    offset = float(np.median((smooth-aligned)[valid]))
    span = max(float(np.ptp(smooth)), float(np.ptp(ref_smooth)), 1e-12)
    residual = np.abs(smooth-aligned-offset)
    # A small coherent residual matters even if the global correlation is high.
    threshold = span*max(.008, recipe['prominence']*.65)
    changed = (residual > threshold) & valid
    components, _ = ndi.label(changed)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    changed = sizes[components] >= max(2, int(round(3*scale*scale)))
    work = changed | ~valid
    work = ndi.binary_dilation(work, iterations=max(1, int(np.ceil(3*scale))))
    predicted = ndi.shift(before['labels'], movement, order=0, mode='constant', cval=0, prefilter=False).astype(np.int32)
    # Repair a whole particle, not just its changed pixels.
    affected = np.unique(predicted[work]); affected = affected[affected > 0]
    work |= np.isin(predicted, affected)
    fraction = float(work.mean())
    info.update(height_offset_nm=offset, residual_threshold_nm=threshold, changed_fraction=fraction)
    if fraction > .35:
        return None, dict(info, reason='Changed or newly visible area is too large for local reuse.')
    ids = set(map(int, np.unique(predicted)))-{0}
    retained = ids-set(map(int, affected))
    labels = np.where(np.isin(predicted, list(retained)), predicted, 0).astype(np.int32)
    records = []
    for particle in before['particles']:
        ident = particle['id']
        if ident in retained:
            records.append(dict(id=ident, uncertain=particle['uncertain'], near_edge=particle['edge'],
                reason=f'Contour reused from frame {base.metadata["frame_0based"]+1} after verified field translation. '+particle['reason']))
    boxes = []
    # Merge nearby changed areas before cropping, preserving the full-field scale
    # and normalization used by the ordinary detector.
    margin = max(4, int(np.ceil(12*scale)))
    repair_area = ndi.binary_dilation(work, iterations=margin)
    objects = _repair_boxes(repair_area)
    if len(objects) > 12:
        return None, dict(info, reason='Too many changed regions for efficient local detection.')
    normalized = (image-image.min())/max(float(np.ptp(image)), 1e-12)
    next_id = max(ids, default=0)+1
    # Maximum LoG sigma in the ordinary detector is 25*scale; crop context must
    # include its support and neighbouring watershed basins.
    context = max(8, int(np.ceil(4*25*scale)))
    crop_pixels = 0
    for y0, x0, y1, x1 in objects:
        check_cancel(cancelled)
        cy0, cy1, cx0, cx1 = max(0, y0-context), min(h, y1+context), max(0, x0-context), min(w, x1+context)
        crop_pixels += (cy1-cy0)*(cx1-cx0)
        if crop_pixels > .85*h*w:
            return None, dict(info, reason='Local detection context covers most of the frame.')
        crop = np.s_[cy0:cy1, cx0:cx1]
        local, candidates, _ = adaptive_watershed_detection(normalized[crop], recipe['prominence'],
            recipe['contour_fraction'], recipe['size_scale'], normalize=False, reference_shape=image.shape)
        inside = np.zeros(local.shape,bool)
        inside[y0-cy0:y1-cy0,x0-cx0:x1-cx0] = work[y0:y1,x0:x1]
        # Select whole detected particles touching a region that needs repair;
        # surrounding particles are context only and retain their anchor masks.
        needed = set(map(int, np.unique(local[inside])))-{0}
        for p in candidates:
            if p['id'] not in needed: continue
            mask = local == p['id']
            # A crop edge must never become an invented particle boundary.
            if ((cy0 and mask[0].any()) or (cy1 < h and mask[-1].any()) or
                    (cx0 and mask[:, 0].any()) or (cx1 < w and mask[:, -1].any()) or
                    np.any(labels[crop][mask])):
                return None, dict(info, reason='A local boundary touches an unchanged particle or crop edge.')
            if next_id > MAX_PARTICLES:
                return None, dict(info, reason='Local particle ID capacity reached.')
            labels[crop][mask] = next_id
            q = deepcopy(p); q['id'] = next_id
            ys, xs = np.nonzero(mask); ys += cy0; xs += cx0
            q['seed_xy'] = [p['seed_xy'][0]+cx0, p['seed_xy'][1]+cy0]
            q['near_edge'] = bool(np.any((xs <= 2*scale) | (xs >= w-1-2*scale) |
                                       (ys <= 2*scale) | (ys >= h-1-2*scale)))
            q['reason'] = 'Local detection in changed or newly visible image area. '+p['reason']
            records.append(q); next_id += 1
        boxes.append([cy0, cx0, cy1, cx1])
    try:
        validate_masks(labels, np.zeros(image.shape, bool), records)
    except ValueError:
        return None, dict(info, reason='Translated masks require complete segmentation.')
    info.update(mode='local' if boxes else 'reused', reason='Verified overlapping field; unchanged contours reused.',
        reused_local_ids=sorted(retained), redetected_regions_yx=boxes, redetected_mask=pack_frame(work),
        reused_particles=len(retained), new_local_particles=len(records)-len(retained))
    chosen = {p['id'] for p in records if not p['uncertain'] and not p['near_edge']}
    result = manual_result(snapshot, labels, dict(method='Adaptive Watershed', detector_recipe=recipe,
        overlap_reuse=info), candidate_records=records, selected_ids=chosen)
    return result, info


def prepare_movie_detections(snapshots, settings=None, cached=None, progress=lambda p,m:None, cancelled=lambda:False,
                             reuse_overlap=False):
    """Prepare local masks once, reusing only identical full-frame inputs/settings.

    Runs in the analysis worker. No Qt controls, AI client, or ASD writes here.
    Cached results are copied by the caller before the worker starts.
    """
    recipe=normal_detection_recipe(settings);results=[];cached=cached or {};reused=0;spatial_reused=0;reference=None
    started=time.monotonic()
    for i,snap in enumerate(snapshots):
        check_cancel(cancelled);source_unchanged(snap)
        index=snap.metadata['frame_0based'];old=cached.get(index);usable=False
        if old is not None:
            before=old['snapshot']
            usable=(old.get('origin')=='manual' and old.get('parameters',{}).get('detector_recipe')==recipe
                and before.metadata.get('coordinate_scope','full-frame')=='full-frame'
                and all(before.metadata.get(k)==snap.metadata.get(k) for k in
                        ('source_path','source_size','source_mtime_ns','frame_0based','channel'))
                and before.pixel_size==snap.pixel_size and np.array_equal(before.image,snap.image))
            if usable:
                try:validate_masks(old['labels'],old['background'],old['particles'])
                except ValueError:usable=False
        if usable:
            result={**deepcopy({k:v for k,v in old.items() if k!='snapshot'}),'snapshot':snap}
            reused+=1;origin='cached normal detection'
        else:
            image=preprocess_normal_detection(snap.image,snap.pixel_size,recipe)
            result=None;overlap_info=None
            if reuse_overlap and reference is not None:
                result,overlap_info=_overlap_detection(reference,image,snap,recipe,cancelled)
            if result is not None:
                spatial_reused+=1;origin='overlapping field: '+overlap_info['mode']
            else:
                labels,records,parameters=adaptive_watershed_detection(image,recipe['prominence'],recipe['contour_fraction'],recipe['size_scale'])
                chosen={p['id'] for p in records if not p['uncertain'] and not p['near_edge']}
                result=manual_result(snap,labels,dict(method='Adaptive Watershed',detector=parameters,detector_recipe=recipe),
                                     candidate_records=records,selected_ids=chosen)
                if reuse_overlap:
                    result['parameters']['overlap_reuse']=dict(overlap_info or {},mode='full',
                        reason=(overlap_info or {}).get('reason','First frame: full detection reference.'))
                reference=(result,image)
                origin='new normal detection'
        if usable and reuse_overlap:
            reference=(result,preprocess_normal_detection(snap.image,snap.pixel_size,recipe))
        result['seed_origin']=origin;results.append(result)
        overlap=result['parameters'].get('overlap_reuse',{})
        detail=(f' {overlap["overlap_fraction"]:.1%} overlap; '+overlap['reason']) if 'overlap_fraction' in overlap else ''
        progress(round(10*(i+1)/max(1,len(snapshots))),f'Normal detection {i+1}/{len(snapshots)}: {origin}; {len(result["particles"])} candidates.'+detail)
    check_cancel(cancelled)
    next_step=('review these detections and repair only defects' if len(snapshots)==1 else 'link IDs and review difficult regions')
    progress(10,f'Normal detection ready: {reused} cached frame(s), {spatial_reused} overlapping frame(s) reused, '
        f'{len(results)-reused-spatial_reused} fully detected ({time.monotonic()-started:.1f} s). AI will {next_step}.')
    return results


def _compact_detection_parameters(result):
    """Keep pixel masks in the audit, not in the agent's JSON text inventory."""
    parameters=deepcopy(result['parameters'])
    parameters.get('overlap_reuse',{}).pop('redetected_mask',None)
    return parameters


def propose_movie_identities(seeds, progress=lambda p,m:None, cancelled=lambda:False):
    """Conservative local ID proposal; ambiguous births/links remain reviewable.

    Reuse measured contours, never regenerate segmentation. Registration searches
    a bounded set of prior observations; matching requires mask, appearance and
    position support, with a margin against competing correspondences.
    """
    from main_view_data import pack_frame
    proposal=[];results=[];next_id=1;doubtful=set()
    for index,seed in enumerate(seeds):
        check_cancel(cancelled);snap=seed['snapshot'];source_unchanged(snap)
        particles={p['id']:p for p in seed['particles']};candidates={k:{} for k in particles}
        registrations=[];visible_prior=set();split_or_merged=set()
        refs=sorted({j for j in (index-1,index-2,index-4,0) if 0<=j<index},reverse=True)
        for ref in refs:
            check_cancel(cancelled);prior=seeds[ref];before=prior['snapshot']
            if snap.pixel_size!=before.pixel_size:continue
            # Identity proposals may span changing fields; unlike frame skipping,
            # this also requires separate contour and local appearance evidence.
            registration,evidence=_overlap_registration(before.image,snap.image,minimum_overlap=.25,minimum_ncc=.75)
            registrations.append(dict(reference_frame_0based=before.metadata['frame_0based'],
                supported=registration is not None,**evidence))
            if registration is None:continue
            movement,valid=registration
            warped=ndi.shift(prior['labels'],movement,order=0,mode='constant',cval=0,prefilter=False)
            old_counts=np.bincount(warped.ravel());new_counts=np.bincount(seed['labels'].ravel())
            smooth=ndi.shift(ndi.gaussian_filter(before.image,.6),movement,order=1,mode='constant',cval=0,prefilter=False)
            current=ndi.gaussian_filter(snap.image,.6)
            old_particles={p['id']:p for p in prior['particles']}
            mapping=proposal[ref]['id_map']
            for local in np.flatnonzero(old_counts[1:])+1:visible_prior.add(mapping[str(local)])
            pairs,counts=np.unique(np.stack((warped.ravel(),seed['labels'].ravel()),axis=1),axis=0,return_counts=True)
            old_to_new={};new_to_old={}
            for (old,local),count in zip(pairs,counts):
                if (old and local and count>=3 and count>=.2*min(old_counts[old],new_counts[local])
                        and count>=.1*max(old_counts[old],new_counts[local])):
                    old_to_new.setdefault(int(old),set()).add(int(local))
                    new_to_old.setdefault(int(local),set()).add(int(old))
            # A split/merge can give one child a strong score. Keep ALL affected
            # observations uncertain instead of counting another child as a birth.
            split_or_merged.update(k for k,v in new_to_old.items() if len(v)>1)
            for children in old_to_new.values():
                if len(children)>1:split_or_merged.update(children)
            for (old,local),count in zip(pairs,counts):
                if not old or not local:continue
                check_cancel(cancelled)
                a=old_particles[int(old)];b=particles[int(local)]
                dice=float(2*count/(old_counts[old]+new_counts[local]))
                ratio=b['area_nm2']/max(a['area_nm2'],1e-30)
                distance=float(np.linalg.norm((np.array([b['y_px']-a['y_px'],b['x_px']-a['x_px']])-movement)*snap.pixel_size[::-1]))
                limit=max(2*max(snap.pixel_size),.35*min(a['equivalent_diameter_nm'],b['equivalent_diameter_nm']))
                if dice<.5 or not .55<=ratio<=1.8 or distance>limit:continue
                left,top,right,bottom=b['bbox_xyxy'];pad=3
                view=np.s_[max(0,top-pad):min(current.shape[0],bottom+pad),max(0,left-pad):min(current.shape[1],right+pad)]
                mask=valid[view];av=smooth[view][mask];bv=current[view][mask]
                if len(av)<8:continue
                av=av-av.mean();bv=bv-bv.mean();denom=float(np.linalg.norm(av)*np.linalg.norm(bv))
                ncc=float(np.dot(av,bv)/denom) if denom>1e-20 else 0.
                if ncc<.85:continue
                score=.5*dice+.3*float(np.clip(ncc,0,1))+.2*(1-distance/limit)
                glob=mapping[str(int(old))]
                evidence=dict(global_id=glob,reference_frame_0based=before.metadata['frame_0based'],
                    reference_local_id=int(old),mask_dice=dice,patch_ncc=ncc,area_ratio=ratio,
                    position_error_nm=distance,score=score)
                if score>candidates[int(local)].get(glob,{}).get('score',-1):candidates[int(local)][glob]=evidence
        ranked={local:sorted(values.values(),key=lambda e:e['score'],reverse=True) for local,values in candidates.items()}
        by_global={}
        for local,values in ranked.items():
            for value in values:by_global.setdefault(value['global_id'],[]).append((value['score'],local))
        mapping={};matches={}
        for local,values in ranked.items():
            if not values or local in split_or_merged:continue
            best=values[0];competitors=sorted(by_global[best['global_id']],reverse=True)
            if (best['score']>=.75 and (len(values)==1 or best['score']-values[1]['score']>=.10)
                    and competitors[0][1]==local and (len(competitors)==1 or competitors[0][0]-competitors[1][0]>=.10)):
                mapping[str(local)]=best['global_id'];matches[str(local)]=best
        unmatched_prior=visible_prior-set(mapping.values())
        unsupported=index>0 and not any(r['supported'] for r in registrations)
        unresolved=sorted(unmatched_prior if not unsupported else {v for p in proposal for v in p['id_map'].values()})
        annotations={};ambiguous=[]
        for local in particles:
            matched=str(local) in mapping
            if not matched:
                if next_id>MAX_PARTICLES:raise ValueError('Movie contains too many provisional particle IDs.')
                mapping[str(local)]=next_id;next_id+=1
            uncertain=bool(index and not matched and (ranked[local] or unresolved or unsupported or local in split_or_merged))
            if uncertain:
                ambiguous.append(local);doubtful.add(mapping[str(local)])
                doubtful.update(unresolved);doubtful.update(v['global_id'] for v in ranked[local])
            reason=('Image registration, contour overlap, shape and local texture support this correspondence.' if matched else
                    'Provisional ID: correspondence or possible duplicate requires review.' if uncertain else
                    'First observation of this particle in the supplied frames.')
            annotations[str(local)]=dict(identity_uncertain=uncertain,reason=reason)
        record=dict(frame_0based=snap.metadata['frame_0based'],id_map=mapping,annotations=annotations,
            registrations=registrations,matches=matches,ambiguous_local_ids=ambiguous,
            split_or_merged_local_ids=sorted(split_or_merged),
            unresolved_prior_ids=unresolved if ambiguous else [],
            alternatives={str(k):v[:3] for k,v in ranked.items() if k in ambiguous})
        proposal.append(record)
        table=np.zeros(int(seed['labels'].max())+1,np.int32)
        for local,glob in mapping.items():table[int(local)]=glob
        labels=table[seed['labels']]
        raw=dict(algorithm='Normal detection + local ID proposal + AI review',
            parameters=dict(identity_method='registered-contour-correspondence-v1',detection_revisions=[]),
            particles=[dict(id=mapping[str(p['id'])],uncertain=p['uncertain'],near_edge=p['edge'],
                **annotations[str(p['id'])]) for p in seed['particles']])
        raw,changes=check_linked_detection(seed,labels,raw)
        # ID-only relabeling cannot change geometry/heights; reuse the host's
        # existing measurements instead of measuring every particle again.
        raw_by_id={p['id']:p for p in raw['particles']};rows=[]
        for p in seed['particles']:
            row=deepcopy(p);row['id']=mapping[str(p['id'])];info=raw_by_id[row['id']]
            row.update(reason=info['reason'],identity_uncertain=info['identity_uncertain'],
                uncertain=info['uncertain'] or info['identity_uncertain'],near_edge=info['near_edge'],
                source_local_ids=info['source_local_ids'],detection_changed=False)
            row.pop('first_frame_1based',None);rows.append(row)
        rows.sort(key=lambda p:p['id'])
        results.append(dict(snapshot=snap,labels=labels,background=seed['background'].copy(),particles=rows,
            origin='ai',algorithm=raw['algorithm'],parameters=raw['parameters'],warnings=list(seed['warnings']),
            summary='',instructions='',ai_accepted=False,audit=dict(raw_result=raw,detection_linking=changes,
                local_identity_proposal=deepcopy(record),normal_detection=dict(labels=pack_frame(seed['labels']),
                    background=pack_frame(seed['background']),particles=deepcopy(seed['particles']),
                    parameters=deepcopy(seed['parameters']),selected_local_ids=sorted(selected_ids(seed)),
                    origin=seed.get('seed_origin','supplied normal detection')))))
        progress(2,f'Local ID proposal: source frame {snap.metadata["frame_0based"]+1}; '
            f'{len(matches)} supported links, {len(ambiguous)} ambiguous IDs.')
    # A possible duplicate flags earlier observations too, so it cannot silently
    # inflate first-appearance statistics even if the AI accepts review flags.
    for record,result in zip(proposal,results):
        for local,glob in record['id_map'].items():
            if glob in doubtful:
                record['annotations'][local]['identity_uncertain']=True
                record['annotations'][local]['reason']+=' A possible duplicate or correspondence elsewhere requires review.'
        result['audit']['local_identity_proposal']=deepcopy(record)
        for group in (result['particles'],result['audit']['raw_result']['particles']):
            for p in group:
                if p['id'] in doubtful:
                    p['identity_uncertain']=True;p['uncertain']=True
                    p['reason']+=' A possible duplicate or correspondence elsewhere requires review.'
    return proposal,results


def check_linked_detection(seed, labels, raw, preserve_uncertainty=True):
    """Validate seed reuse independent of global ID numbering and audit local edits.

    Unchanged shapes need only an ID map. A deleted, added, split, merged or
    resized region must be covered by declared half-open NumPy bounding boxes.
    The host measures the edits rather than trusting an agent's reuse claim.
    """
    original=seed['labels'];h,w=original.shape
    revisions=raw['parameters'].get('detection_revisions',[])
    if not isinstance(revisions,list) or len(revisions)>MAX_PARTICLES:raise ValueError('Invalid detection revision list.')
    allowed=np.zeros(original.shape,bool);clean=[]
    for revision in revisions:
        box=revision.get('bbox_yx') if isinstance(revision,dict) else None
        reason=revision.get('reason') if isinstance(revision,dict) else None
        if (not isinstance(box,list) or len(box)!=4 or any(type(v)is not int for v in box)
                or not 0<=box[0]<box[2]<=h or not 0<=box[1]<box[3]<=w
                or not isinstance(reason,str) or not reason.strip() or len(reason)>4000):
            raise ValueError('Each local detection revision needs bbox_yx=[top,left,bottom,right] and a measured reason.')
        allowed[box[0]:box[2],box[1]:box[3]]=True
        clean.append(dict(bbox_yx=box,reason=reason.strip()))
    # Count intersections once, including background. IDs can be sparse.
    # np.unique's pair table avoids O(number_of_particles * image_pixels) masks.
    pairs,counts=np.unique(np.stack((original.ravel(),labels.ravel()),axis=1),axis=0,return_counts=True)
    old_counts=np.bincount(original.ravel());new_counts=np.bincount(labels.ravel())
    mapping={int(a):int(b) for (a,b),count in zip(pairs,counts)
             if a and b and count==old_counts[a] and count==new_counts[b]}
    changed=((original>0)&~np.isin(original,list(mapping))) | ((labels>0)&~np.isin(labels,list(mapping.values())))
    if np.any(changed & ~allowed):
        raise ValueError('AI changed normal detection outside declared local revision boxes. Preserve masks and only relabel IDs, '
                         'or record detection_revisions with boxes covering each entire changed old/new particle and a measured reason.')
    raw=deepcopy(raw);seed_records={p['id']:p for p in seed['particles']};inverse={v:k for k,v in mapping.items()}
    overlaps={int(i):[] for i in new_counts.nonzero()[0] if i}
    for (a,b),count in zip(pairs,counts):
        if a and b:overlaps[int(b)].append(int(a))
    for p in raw['particles']:
        local=inverse.get(p['id']);p['source_local_ids']=overlaps[p['id']]
        p['detection_changed']=local is None
        if local is not None:
            prior=seed_records[local]
            p['uncertain']=p['uncertain'] or (preserve_uncertainty and prior['uncertain'])
            p['near_edge']=p.get('near_edge',False) or (preserve_uncertainty and prior['edge'])
            p['reason']=f'Normal detection #{local} reused; contour unchanged. '+p['reason']
        else:
            reasons=[r['reason'] for r in clean if np.any(labels[r['bbox_yx'][0]:r['bbox_yx'][2],
                r['bbox_yx'][1]:r['bbox_yx'][3]]==p['id'])]
            p['reason']='Local detection revised; review contour. '+' '.join(reasons)+' '+p['reason']
    audit=dict(unchanged_id_map={str(k):v for k,v in mapping.items()},reused_particles=len(mapping),
               revised_particles=len(raw['particles'])-len(mapping),changed_pixels=int(changed.sum()),revisions=clean)
    return raw,audit


MOVIE_SUPPORT = '''from pathlib import Path
import json
import numpy as np
from copy import deepcopy
from functools import lru_cache
from analysis_support import save, load_checkpoint

@lru_cache(maxsize=1)
def load_movie():
    return np.load('frames.npy',mmap_mode='r',allow_pickle=False),json.loads(Path('movie_metadata.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=1)
def load_detections():
    return np.load('normal_labels.npy',mmap_mode='r',allow_pickle=False),json.loads(Path('normal_detections.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=1)
def load_identity_proposal():
    return json.loads(Path('identity_proposal.json').read_text(encoding='utf-8'))

def save_proposed_frame(index, id_map=None, annotations=None, parameters=None, background=None):
    proposal=load_identity_proposal()['frames'][index]
    notes=deepcopy(proposal['annotations'])
    for key,value in (annotations or {}).items():notes.setdefault(str(key),{}).update(value)
    params=dict(parameters or {})
    params.setdefault('identity_method','registered-contour-correspondence-v1')
    if background is None:background=np.load('normal_background.npy',mmap_mode='r',allow_pickle=False)[index]
    save_linked_frame(index,proposal['id_map'] if id_map is None else id_map,notes,
        'Normal detection + reviewed ID proposal',params,background)

def save_checkpoint_frame(index, version=1):
    labels,background,records=load_checkpoint(version)
    raw=records[index]
    save_frame(index,labels[index],background[index],raw['particles'],raw['algorithm'],raw['parameters'],raw['warnings'])

def save_checkpoint_movie(version=1):
    _,meta=load_movie()
    for index in range(meta['frame_count']):save_checkpoint_frame(index,version)

def save_linked_frame(index, id_map, annotations=None, algorithm='Normal detection + AI ID linking', parameters=None, background=None):
    seeds,meta=load_detections();labels=seeds[index];mapping={int(k):int(v) for k,v in id_map.items()}
    ids=set(np.unique(labels))-{0}
    if set(mapping)!=ids or len(set(mapping.values()))!=len(mapping) or any(not 1<=v<=10000 for v in mapping.values()):
        raise ValueError('Map every local ID to one distinct positive movie-global ID in this frame.')
    table=np.zeros(int(labels.max())+1,dtype=np.int32)
    for k,v in mapping.items():table[k]=v
    particles=[];annotations=annotations or {}
    for prior in meta['frames'][index]['particles']:
        local=prior['id'];note=annotations.get(local,annotations.get(str(local),{}))
        particles.append(dict(id=mapping[local],uncertain=bool(prior['uncertain']),near_edge=bool(prior['edge']),
            identity_uncertain=bool(note.get('identity_uncertain',False)),
            reason=str(note.get('reason','Normal detection reused; correspondence from the recorded linking method.'))))
    params=dict(parameters or {});params['detection_revisions']=[]
    save_frame(index,table[labels],np.zeros(labels.shape,bool) if background is None else background,particles,algorithm,params)

def save_frame(index, labels, background, particles, algorithm, parameters=None, warnings=None):
    import os
    root=Path.cwd();folder=root/'results'/f'{int(index):06d}';folder.mkdir(parents=True,exist_ok=True)
    _,meta=load_movie()
    # The shared single-frame helper writes an audited frame contract.
    np.save(folder/'frame.npy',load_movie()[0][index],allow_pickle=False)
    import hashlib
    record=dict(meta['frame_metadata'][index],input_sha256=hashlib.sha256((folder/'frame.npy').read_bytes()).hexdigest())
    (folder/'metadata.json').write_text(json.dumps(record))
    try:
        os.chdir(folder);save(labels,background,particles,algorithm,parameters,warnings)
    finally:os.chdir(root)
'''


def movie_sheet(snapshots, results, indices, target):
    from PIL import Image,ImageDraw
    width=360;tile_h=390;sheet=Image.new('RGB',(width*3,tile_h*math.ceil(len(indices)/3)),'#181818')
    draw=ImageDraw.Draw(sheet)
    for k,index in enumerate(indices):
        rows=results[index]['particles'] if results else ()
        tile=render(snapshots[index],rows,long_edge=340)
        x,y=(k%3)*width,(k//3)*tile_h
        sheet.paste(tile,(x+(width-tile.width)//2,y+30+(350-tile.height)//2))
        draw.text((x+8,y+8),f'Frame {snapshots[index].metadata["frame_0based"]+1}',fill='white')
    target=Path(target)
    with open_host_output(target.parent,target.name) as stream: sheet.save(stream,format='PNG')


@bounded_analysis
def run_movie_analysis(snapshots, client, workspace, instructions='', progress=lambda p,m:None,
                       cancelled=lambda:False, time_limit=900,max_rounds=2, initial_detections=None, frame_sampling=None,
                       refinement=None):
    """Review a local ID proposal first; generate repair code only when needed.

    Does not make one full AI request per frame. All frames are numerically
    checked; contact sheets cover sampled frames and uncertain extremes.
    """
    if not snapshots:raise ValueError('No frames to analyze.')
    if any(s.image.shape!=snapshots[0].image.shape for s in snapshots):raise ValueError('Movie shape mismatch.')
    if frame_sampling is not None and [p['frame_0based'] for p in frame_sampling if p['status']=='analyzed']!=[s.metadata['frame_0based'] for s in snapshots]:
        raise ValueError('Representative frames do not match the overlap screening plan.')
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    if type(max_rounds) is not int or not 1<=max_rounds<=12: raise ValueError('AI rounds must be between 1 and 12.')
    budget=current_budget();nframes=len(snapshots)
    def check(check_time=True):
        check_cancel(cancelled);source_unchanged(snapshots[0])
        budget.remaining(check_time)
    percent=0;timings=[]
    def report(p,msg):
        nonlocal percent
        check();percent=p;progress(p,msg)
    def timed(stage,operation):
        started=time.monotonic()
        try:
            result=operation();budget.touch();return result
        finally:
            seconds=round(time.monotonic()-started,3)
            timings.append(dict(stage=stage,seconds=seconds));progress(percent,f'{stage}: {seconds:.1f} s.')
    check()
    movie=np.lib.format.open_memmap(root/'frames.npy',mode='w+',dtype=np.float64,shape=(nframes,*snapshots[0].image.shape))
    for index,snap in enumerate(snapshots):check();movie[index]=snap.image;budget.touch()
    movie.flush();del movie
    frame_hashes=[]
    for snap in snapshots:
        stream=io.BytesIO();np.save(stream,snap.image,allow_pickle=False)
        import hashlib
        frame_hashes.append(hashlib.sha256(stream.getvalue()).hexdigest())
    meta=dict(frame_count=nframes,shape=list(snapshots[0].image.shape),pixel_size_nm=list(snapshots[0].pixel_size),
        height_unit='nm',coordinates='NUMPY row/column, NOT screen coordinates. Host preview is vertically flipped.',
        frame_metadata=[dict(frame_0based=s.metadata['frame_0based']) for s in snapshots])
    if frame_sampling is not None:
        meta['frame_sampling']=dict(analyzed_source_frames_1based=[s.metadata['frame_0based']+1 for s in snapshots],
            skipped_source_frames_1based=[p['frame_0based']+1 for p in frame_sampling if p['status']=='skipped'],
            overlap_threshold=FIELD_OVERLAP_MIN)
    (root/'movie_metadata.json').write_text(json.dumps(meta));(root/'analysis_support.py').write_text(SUPPORT_CODE)
    (root/'movie_support.py').write_text(MOVIE_SUPPORT)
    seeded=initial_detections is not None
    movie_start_mode=(initial_detections[0]['audit'].get('detection_start_mode','default_detection') if refinement is not None
                      else 'default_detection' if seeded else 'ai_from_scratch')
    seed_files=[]
    if seeded:
        if len(initial_detections)!=nframes:raise ValueError('Normal detections must cover every movie frame.')
        seeds=np.lib.format.open_memmap(root/'normal_labels.npy',mode='w+',dtype=np.int32,shape=(nframes,*snapshots[0].image.shape))
        backgrounds=np.lib.format.open_memmap(root/'normal_background.npy',mode='w+',dtype=bool,shape=seeds.shape)
        items=[]
        for i,(snap,result) in enumerate(zip(snapshots,initial_detections)):
            check();before=result['snapshot']
            if (snap.metadata['frame_0based']!=before.metadata['frame_0based'] or snap.pixel_size!=before.pixel_size
                    or not np.array_equal(snap.image,before.image) or result['labels'].shape!=snap.image.shape):
                raise ValueError('Normal detection does not match the captured main frame.')
            validate_masks(result['labels'],result['background'],result['particles']);seeds[i]=result['labels']
            backgrounds[i]=result['background']
            items.append(dict(frame_0based=snap.metadata['frame_0based'],origin=result.get('seed_origin','supplied normal detection'),
                parameters=_compact_detection_parameters(result),selected_local_ids=sorted(selected_ids(result)),
                particles=[{k:p[k] for k in ('id','x_px','y_px','area_nm2','equivalent_diameter_nm','major_axis_nm','minor_axis_nm',
                                             'mean_z_nm','max_z_nm','uncertain','edge','reason')} for p in result['particles']]))
        seeds.flush();backgrounds.flush();del seeds,backgrounds
        (root/'normal_detections.json').write_text(json.dumps(dict(
            id_scope=refinement['request']['id_scope'] if refinement else 'frame-local',frames=items),allow_nan=False))
        seed_files=['normal_labels.npy','normal_background.npy','normal_detections.json','normal_overview.png']
    indices=sorted(set(np.linspace(0,nframes-1,min(9,nframes)).astype(int).tolist()))
    if refinement is not None:
        focus=next(i for i,s in enumerate(snapshots) if s.metadata['frame_0based']+1==refinement['request']['frame_1based'])
        indices=sorted(set(indices+list(range(max(0,focus-1),min(nframes,focus+2)))))
    if seeded:
        references=[i for i,r in enumerate(initial_detections) if r['parameters'].get('overlap_reuse',{}).get('mode')=='full']
        repairs=[i for i,r in enumerate(initial_detections) if r['parameters'].get('overlap_reuse',{}).get('mode')=='local']
        indices=sorted(set(indices+references[:3]+repairs[:3]))
    movie_sheet(snapshots,None,indices,root/'input_overview.png')
    if seeded:movie_sheet(snapshots,initial_detections,indices,root/'normal_overview.png')
    immutable={n:digest(root/n) for n in ['frames.npy','movie_metadata.json','analysis_support.py','movie_support.py','input_overview.png']+seed_files}
    if refinement is not None:
        prepare_refinement_inputs(root, refinement, initial_detections, immutable, movie=True)
    def verify(check_time=True):
        check(check_time)
        if any((root/n).is_symlink() or digest(root/n)!=v for n,v in immutable.items()):raise ValueError('AI changed its supplied movie input; rejected.')
    command=worker_command(root)
    identity_rule=('Assign MOVIE-GLOBAL particle IDs after inspecting ALL frames. The same physical particle must retain '
        'the same positive label ID in every observation, including supported disappearance/reappearance. '
        'Detect new particles, but do not count a moving or returning particle as new. Optimize correspondence for these data '
        'using drift/field changes, shape, neighborhood and temporal evidence; do not rely on nearest position alone. '
        'Never reuse an ID for a different particle or force a match between unrelated fields. '
        'Flag identity_uncertain=true for every affected ID when correspondence, an ID swap or a possible duplicate remains ambiguous. '
        'Keep an identity uncertainty reason and measured linking evidence/parameters in the output. '
        'Statistics use ONLY each ID\'s FIRST appearance; later images never replace its measurements.')
    if frame_sampling is not None:
        identity_rule+=(' This is a representative-frame sample, NOT every source frame. ALL/EVERY frame below means '
            'every supplied array index. Source frame numbers are in movie_metadata.json frame_metadata. '
            'Overlapping source frames were deliberately skipped by the user option; their images are not supplied. '
            'Do not reconstruct, segment or measure skipped frames. Statistics mean FIRST ANALYZED appearance; '
            'do not claim exact first appearance or continuous identity through skipped intervals. '
            'Link retained observations using available evidence and flag ambiguous matches across gaps.')
    payload=dict(task='Detect and segment particles throughout this AFM movie. '+identity_rule+' '
        'Inspect the data, compare methods on representative and difficult frames, write one analysis.py that adapts to frame variation '
        'and processes EVERY frame. Do not request separate AI analysis for each frame. Run the script, inspect overlays, revise as needed.',
        metadata=meta,runtime_command=command,user_instructions=instructions,
        interface='from movie_support import load_movie, save_frame; frames,meta=load_movie() returns [frame,row,column] heights. '
            'Call save_frame(index,labels,background,particles,algorithm,parameters,warnings) for every index 0..frame_count-1. '
            'Each labels array is integer with 0 background and positive unique IDs <=10000, shape equal to the input frame. '
            'Each connected particle has {id:int, uncertain:bool, identity_uncertain:bool, reason:str}. Background is a bool mask of supported substrate, disjoint from all particles. '
            'Use all-False background if uncertain. Record actual algorithm and parameters.',
        requirements=['Input is already processed by the main window. Do not double-level or change input heights. '
            'Detection-only smoothing is allowed; host measures original numerical input.',
            'Use data-dependent scale/threshold/segmentation, preserve plausible weak and edge candidates for human choice. '
            'Do not invent coordinates/counts or fill watershed basins with substrate. Compare parameter stability and touching-object separation.',
            'Keep labels in original array orientation. The overview is flipped for display only.',
            'Preserve helpers and inputs. Work only in this workspace, no packages, network or application edits. '
            'Regenerate outputs from data deterministically; no background commands. All frames must be represented, even zero-particle frames.'])
    seed_rule=('Inspect the movie data and use supplied NORMAL DETECTION as reference segmentation. All its IDs are FRAME-LOCAL, '
        'not identities across frames. Focus computation on data-dependent drift-aware ID correspondence, appearance/disappearance '
        'and ambiguous intervals. Preserve correct contours and weak/edge candidates. Choose algorithms for THIS data. '
        'For local defects, re-detect only affected regions. If the reference method systematically fails for this movie '
        'or a different field, compare alternatives on representative frames and replace the algorithm on the affected '
        'frames/regions. Record parameters.method_reassessment with observed failures, compared methods and chosen method. '
        'Do not repeat equivalent segmentation or parameter sweeps on every frame; cached results are measurements to reuse '
        'when valid, not a restriction against changing an unsuitable detector. '

        'When parameters.overlap_reuse is present, the host has checked every frame against a fully detected reference. '
        'A reused/local frame has at least 90% field overlap plus verified texture alignment. '
        'reused_local_ids are unchanged anchor contours translated by translation_yx_px; use them and '
        'reference_frame_0based as efficient proposed ID links to that anchor. Newly detected local IDs are not identities. '
        'Prioritize new/changed regions and ambiguous links rather than repeatedly redesigning detection for stable shared areas. '
        'Spatial overlap is not proof of molecular identity: still check motion, replacement, disappearance and new arrivals. '

        'Every changed old/new particle must lie inside an explicitly recorded detection_revisions box; an ID-only change needs no revision. '
        'Do not declare whole-frame or whole-movie revisions merely to bypass reuse. Explain the observed defect for every revision.')
    if seeded:
        payload['task']='Choose suitable segmentation and link particle identities throughout this AFM movie, reusing correct reference detections. '+identity_rule+' '+seed_rule+' Write one analysis.py and validate the measured result.'
        payload['interface']+=(' For normal masks, use load_detections() from movie_support: returns a read-only [frame,row,column] '
            'label array and metadata with per-frame particles, parameters and selected_local_ids. These selections are for human review, '
            'NOT permission to discard candidates. save_linked_frame(index, id_map, annotations, algorithm, parameters, background=None) '
            'preserves every mask exactly and assigns movie IDs. id_map maps ALL positive local IDs to distinct global IDs; '
            'annotations is keyed by local ID with reason and identity_uncertain. Empty frames use {}. '
            'For a local re-segmentation use save_frame; parameters must include detection_revisions: '
            '[{bbox_yx:[top,left,bottom,right],reason:"measured problem and repair"}]. Boxes use half-open NUMPY coordinates '
            'and cover the ENTIRE old and new regions being changed. Unchanged frames use an empty list. '
            'Outside these boxes the host requires the exact original particle shapes, irrespective of global ID numbering.')
        payload['requirements'][1]=seed_rule
        payload['normal_detection_counts']=[len(r['particles']) for r in initial_detections]
        payload['overlap_reuse_summary']={mode:sum(r['parameters'].get('overlap_reuse',{}).get('mode')==mode
            for r in initial_detections) for mode in ('full','reused','local')}
    reports=[];last=None;last_error=''
    retained_movie=copy_results(initial_detections) if refinement is not None else None
    def retain_measured(message):
        verify(check_time=False)
        if retained_movie is None:
            from ai_recovery import TotalTimeLimit
            raise TotalTimeLimit(message)
        retained=incomplete_measurements(retained_movie, message, keep_flags=refinement is not None,
            audit=dict(detection_start_mode=movie_start_mode,execution_mode='retained_at_total_limit',
                       provider=client.settings.get('provider','codex'),model=getattr(client,'model',client.settings.get('model'))))
        for result in retained: result['instructions']=instructions
        return make_session(retained,'all','movie-global')
    budget.checkpoint=retain_measured
    measured_previous=initial_detections
    checkpoint_version=None
    proposal=None
    if refinement is not None:
        seed_rule=seed_rule.replace('All its IDs are FRAME-LOCAL, not identities across frames.',
            'Its IDs already use the id_scope in refinement.json. Preserve supported movie-global identities.')
        payload['task']=REFINEMENT_RULE+' '+identity_rule+' '+seed_rule+' Write analysis.py and validate the changed results.'
        payload['refinement']=refinement['request']
        payload['previous_result_file']='refinement.json'
        payload['interface']+=(' Start analysis.py with save_checkpoint_movie(0) from movie_support to restore the current '
            'result. load_checkpoint(0) from analysis_support returns all masks, backgrounds and raw metadata. '
            'Overwrite only corrected frames with save_frame. Preserve unchanged IDs and review flags.')
        payload['requirements'][1]=seed_rule
        checkpoint_version=0
    elif seeded:
        report(2,'Building a local particle-ID proposal from the existing contours…')
        proposal,proposed_results=timed('Local ID proposal',lambda:propose_movie_identities(initial_detections,report,cancelled))
        retained_movie=copy_results(proposed_results)
        proposal_doc=dict(format='pynud-particle-identity-proposal-v1',frames=proposal,
            limitations='A bounded registration and contour-matching proposal, not proof of molecular identity. '
                'Unsupported correspondence and possible duplicates remain flagged for human review.',
            parameters=dict(reference_offsets=[1,2,4],also_first_frame=True,minimum_field_overlap=.25,
                registration_ncc=.75,registration_peak_ratio=1.03,minimum_mask_dice=.5,
                area_ratio_range=[.55,1.8],minimum_patch_ncc=.85,minimum_match_score=.75,ambiguity_margin=.10))
        (root/'identity_proposal.json').write_text(json.dumps(proposal_doc,allow_nan=False))
        proposal_files=['identity_proposal.json']
        difficult=[i for i,r in enumerate(proposed_results) if any(p['identity_uncertain'] for p in r['particles'])]
        reviewed=sorted(set(indices+difficult[:3]+[max(0,i-1) for i in difficult[:3]]))
        images=[]
        for start in range(0,len(reviewed),9):
            group=reviewed[start:start+9]
            for name,rows in ((f'proposal_input_{start}.png',None),(f'proposal_result_{start}.png',proposed_results)):
                movie_sheet(snapshots,rows,group,root/name);images.append(str(root/name));proposal_files.append(name)
        immutable.update({name:digest(root/name) for name in proposal_files})
        report(4,'AI reviewing normal contours and the local ID proposal; no analysis code is needed unless a defect is found…')
        client.settings['timeout']=max(1,int(budget.remaining()))
        review=check_reply(timed('AI local-ID proposal review',lambda:client.run_agent(dict(
            task='Review the supplied normal segmentation and HOST-COMPUTED particle-ID proposal. '+identity_rule+' '+seed_rule+' '
                'Use the provided overlays and recorded correspondence evidence in identity_proposal.json. '
                'This is a bounded review, not a new numerical analysis: do NOT write or execute code, rerun detection, '
                'sweep parameters, or independently recompute the full movie. Inspect the recorded uncertainties and '
                'source-frame gaps, and check representative shapes/links visually. Accept if the proposal meets the user instructions '
                'with appropriate uncertainty flags for human review. Acceptance preserves masks, measurements, IDs and flags exactly. '
                'Do not accept blanket uncertainty flags instead of resolving clearly supported correspondences. '
                'If most identities are uncertain, request targeted repair unless these images genuinely cannot support correspondence. '
                'Request revision for concrete wrong IDs, unsupported confident matches, missed/merged particles, or user-requested '
                'repairs/background measurements; name the source frames/IDs and required changes. '
                'Do not claim every particle or unsupplied frame was visually verified.',
            metadata=meta,user_instructions=instructions,proposal_file='identity_proposal.json',
            normal_detection_file='normal_detections.json',
            frame_summary=[dict(frame_1based=r['snapshot'].metadata['frame_0based']+1,particles=len(r['particles']),
                supported_links=len(p['matches']),identity_uncertain=sum(row['identity_uncertain'] for row in r['particles']))
                for r,p in zip(proposed_results,proposal)],
            visually_supplied_frames_1based=[snapshots[i].metadata['frame_0based']+1 for i in reviewed],
            overlap_reuse_summary=payload['overlap_reuse_summary']),images,result_schema(True),
            lambda m:report(4,m),writable=False)),True)
        reports.append(review);verify()
        if review['decision']=='accept':
            for result in proposed_results:
                result.update(summary=review['summary'],instructions=instructions,ai_accepted=True)
                result['warnings']=list(dict.fromkeys(result['warnings']+review['warnings']))
                if any(p['identity_uncertain'] for p in result['particles']):
                    result['warnings'].append('Possible duplicate/identity ambiguity retained; review purple particles before including them in statistics.')
                result['audit'].update(metadata=meta,reports=deepcopy(reports),provider=client.settings.get('provider', 'codex'),
                    model=getattr(client,'model',client.settings.get('model')),
                    reasoning_effort=getattr(client,'reasoning_effort',None),execution_mode='local-id-proposal-review',
                    detection_start_mode='default_detection',
                    ai_request_count=1,stage_timings=deepcopy(timings),numerical_frame_count=nframes,
                    visual_review_frames_1based=[snapshots[i].metadata['frame_0based']+1 for i in reviewed],
                    proposal_parameters=deepcopy(proposal_doc['parameters']))
                if frame_sampling is not None:result['audit']['sampling_summary']=deepcopy(meta['frame_sampling'])
            verify();report(100,f'AI accepted the local ID proposal for {nframes} frames: one review request, no generated analysis code.')
            return make_session(proposed_results,'all','movie-global')
        payload['task']='Repair only the concrete problems in the local ID proposal identified by the review. '+identity_rule+' '+seed_rule+' Write analysis.py and validate the changed results.'
        payload['initial_identity_review']=review
        payload['identity_proposal_file']='identity_proposal.json'
        payload['interface']+=(' from movie_support import load_identity_proposal, save_proposed_frame. '
            'load_identity_proposal()["frames"][index] contains the host id_map, annotations, registration and matching evidence. '
            'Call save_proposed_frame(index) for unchanged frames. Supply id_map / annotations overrides only for needed ID repairs, '
            'including all affected observations and first appearances. Use save_frame only for justified local contour repairs. '
            'Avoid recomputing supported correspondences and detections; prioritize the review and user instructions.')
        report(5,'AI requested targeted repair of the ID proposal: '+review['summary'])
    payload.setdefault('requirements',[]).append(REPLAY_RULE)
    base_payload=deepcopy(payload)
    for attempt in range(max_rounds):
        check();client.settings['timeout']=max(1,int(budget.remaining()));pct=5+attempt*28
        report(pct,(f'AI linking normal detections across {nframes} frames' if seeded else f'AI optimizing detection across {nframes} frames')+f' — round {attempt+1}/{max_rounds}…')
        input_images=[str(root/'input_overview.png')]+([str(root/'normal_overview.png')] if seeded else [])
        answer=check_reply(timed('AI targeted repair' if seeded else 'AI movie analysis',lambda:client.run_agent(payload,input_images,result_schema(),lambda m:report(pct,m))))
        reports.append(answer);verify()
        try:
            script=read_workspace_file(root,'analysis.py',1024*1024)
            # Stale per-frame files are never accepted as a new measured result.
            import shutil
            output=root/'results'
            if output.is_symlink():raise ValueError('Result folder must not be a link.')
            if output.exists():shutil.rmtree(output)
            with isolated_replay(root,immutable,discard=('results',)):
                timed('Sandbox script verification',lambda:client.execute_analysis(command,timeout=replay_timeout(client,(nframes,*snapshots[0].image.shape))))
            verify()
            if output.is_symlink():raise ValueError('Result folder must not be a link.')
            results=[]
            for index,snap in enumerate(snapshots):
                check();folder=output/f'{index:06d}'
                if folder.is_symlink():raise ValueError('Invalid frame result folder.')
                labels,bg,raw=validate_segmentation(folder,snap,frame_hashes[index])
                audit=dict(raw_result=deepcopy(raw))
                if seeded:
                    from main_view_data import pack_frame
                    raw,changes=check_linked_detection(initial_detections[index],labels,raw,preserve_uncertainty=refinement is None)
                    seed=initial_detections[index]
                    audit.update(normal_detection=dict(labels=pack_frame(seed['labels']),particles=deepcopy(seed['particles']),
                        selected_local_ids=sorted(selected_ids(seed)),parameters=deepcopy(seed['parameters']),
                        origin=seed.get('seed_origin','supplied normal detection')),
                        detection_linking=changes)
                    progress(pct,f'Frame {snap.metadata["frame_0based"]+1}: {changes["reused_particles"]} contours reused; {changes["revised_particles"]} revised; {len(changes["revisions"])} local repair region(s).')
                prior=measured_previous[index] if measured_previous else None
                rows,reused=measure_reusing_previous(snap,labels,bg,raw,prior,cancelled)
                audit['measurements_reused']=reused
                if prior is not None:
                    audit['checkpoint_inputs']=deepcopy(prior.get('audit',{}).get('checkpoint_inputs',{}))
                results.append(dict(snapshot=snap,labels=labels,background=bg,particles=rows,origin='ai',
                    algorithm=raw['algorithm'],parameters=raw['parameters'],warnings=raw['warnings'],summary='',
                    instructions=instructions,ai_accepted=False,audit=audit))
                action='Reusing geometry/heights for' if reused else 'pyNuD measuring'
                progress(pct+round(12*(index+1)/nframes),f'{action} source frame {snap.metadata["frame_0based"]+1} ({index+1}/{nframes})…')
        except (ValueError, OSError, KeyError, CodexError) as exc:
            if isinstance(exc, TimeoutError) or is_timeout_message(exc): raise
            check();last_error=str(exc)
            payload={**base_payload,'task':'Repair only the validation failure. Restore unchanged outputs from the latest '
                'checkpoint using save_checkpoint_movie(version), then overwrite repaired frames through save_frame. '
                'If no checkpoint exists, use the normal detections/ID proposal for unchanged frames. '
                +identity_rule+(' '+seed_rule if seeded else ''),'error':last_error,'latest_checkpoint':checkpoint_version}
            report(pct,'Output needs repair: '+last_error)
            continue
        measured_previous=results
        retained_movie=copy_results(results)
        for result in retained_movie:
            result['audit'].update(script=script,script_sha256=digest(root/'analysis.py'),
                support_script=SUPPORT_CODE,movie_support_script=MOVIE_SUPPORT,reports=deepcopy(reports),
                provider=client.settings.get('provider','codex'),numerical_frame_count=nframes)
        checkpoint_version=attempt+1
        save_measured_checkpoint(root,checkpoint_version,results,immutable,movie=True)
        # Extra examples target the largest uncertainty count and count changes.
        ranked=sorted(range(nframes),key=lambda i:sum(p['uncertain'] for p in results[i]['particles']),reverse=True)
        counts=[len(r['particles']) for r in results]
        jumps=sorted(range(1,nframes),key=lambda i:abs(counts[i]-counts[i-1]),reverse=True)
        reviewed=sorted(set(indices+ranked[:3]+jumps[:3]))
        images=list(input_images)
        for start in range(0,len(reviewed),9):
            group=reviewed[start:start+9];name=root/f'measured_{start}.png';original=root/f'original_{start}.png'
            movie_sheet(snapshots,results,group,name);movie_sheet(snapshots,None,group,original)
            images.extend([str(original),str(name)])
        hashes={str(p.relative_to(root)):digest(p) for p in output.rglob('*') if p.is_file()}
        hashes['analysis.py']=digest(root/'analysis.py')
        identities={}
        for i,r in enumerate(results):
            for p in r['particles']:
                identities.setdefault(str(p['id']),[]).append(dict(frame=r['snapshot'].metadata['frame_0based']+1,x=p['x_px'],y=p['y_px'],
                    area_nm2=p['area_nm2'],identity_uncertain=p['identity_uncertain']))
        write_host_text(root,'measured_identities.json',json.dumps(identities))
        changes=[dict(frame=r['snapshot'].metadata['frame_0based']+1,**r['audit']['detection_linking']) for r in results] if seeded else []
        write_host_text(root,'detection_changes.json',json.dumps(changes))
        hashes['measured_identities.json']=digest(root/'measured_identities.json')
        hashes['detection_changes.json']=digest(root/'detection_changes.json')
        report(pct+18,'AI reviewing measured contours, particle identities and duplicate counts…')
        client.settings['timeout']=max(1,int(budget.remaining()))
        client.progress=lambda m:report(pct+18,m)
        review=review_measurements(lambda:timed('AI measured-result review',lambda:client.advise(dict(task='Review the measured segmentation across the movie. '
            'Check weak omissions, merged/split particles, changing counts, orientation and contour boundaries. '+identity_rule+' '
            'Use the supplied measured identity trajectories to check births, reappearances and ambiguous links. '
            'Do not execute code, open files or repeat numerical analysis. '
            'All frames were numerically validated; do not claim visual review of unsampled frames. Request revision if needed.'+
            (' '+seed_rule+' Compare the normal overview with measured results and the supplied detection_changes.' if seeded else ''),
            refinement_request=refinement['request'] if refinement else None,
            measured_identities=identities,detection_changes=changes,frame_particle_counts=counts,visually_supplied_frames_1based=[snapshots[i].metadata['frame_0based']+1 for i in reviewed],
            methods=[dict(frame=snapshots[i].metadata['frame_0based']+1,algorithm=results[i]['algorithm'],parameters=results[i]['parameters']) for i in indices],
            user_instructions=instructions),images,result_schema(True))),verify)
        verify();reports.append(review)
        if any((root/p).is_symlink() or digest(root/p)!=value for p,value in hashes.items()):raise ValueError('Outputs changed during read-only review.')
        for result in results:
            result['summary']=review['summary'];result['ai_accepted']=review['decision']=='accept'
            result['warnings']=list(dict.fromkeys(result['warnings']+answer['warnings']+review['warnings']))
            result['audit'].update(script=script,script_sha256=hashes['analysis.py'],support_script=SUPPORT_CODE,
                movie_support_script=MOVIE_SUPPORT,reports=deepcopy(reports),provider=client.settings.get('provider', 'codex'),model=getattr(client,'model',client.settings.get('model')),
                numerical_frame_count=nframes,visual_review_frames_1based=[snapshots[i].metadata['frame_0based']+1 for i in reviewed],
                execution_mode='refinement' if refinement is not None else 'targeted-repair' if seeded else 'agent-analysis',ai_request_count=len(reports),
                detection_start_mode=movie_start_mode,
                stage_timings=deepcopy(timings))
            if frame_sampling is not None:result['audit']['sampling_summary']=deepcopy(meta['frame_sampling'])
        if proposal is not None:
            for result,initial in zip(results,proposal):result['audit']['local_identity_proposal']=deepcopy(initial)
        last=results
        retained_movie=last
        if review.get('review_incomplete'):
            for result in last:
                result['audit']['ai_review_status']='incomplete'
                if refinement is None:
                    for p in result['particles']:p['uncertain']=True
            break
        if review['decision']=='accept':break
        payload={**base_payload,'task':f'Preserve the latest verified movie with save_checkpoint_movie({checkpoint_version}) '
            'from movie_support at the start of analysis.py. Then repair only the affected frames/regions and overwrite them '
            'with save_frame. load_checkpoint(version) from analysis_support provides masks, backgrounds and raw metadata '
            'for all frames; copy arrays before changing them. For ID-only changes, reuse masks and measurements. '
            'Propagate ID corrections to all affected observations without re-detecting good contours. Make one focused '
            'repair pass, execute the script and inspect the changed results. '+identity_rule+(' '+seed_rule if seeded else ''),
            'review':review,'latest_checkpoint':checkpoint_version}
    else:
        if last is None:raise ValueError('No complete valid movie segmentation: '+last_error)
        for result in last:
            result['ai_accepted']=False;result['warnings'].append('Unfinished AI revision; all candidates require review.')
            if refinement is None:
                for p in result['particles']:p['uncertain']=True
    check();verify();report(100,f'Completed {nframes} frames. Review segmentations and particle IDs before export.')
    return make_session(last,'all','movie-global')
