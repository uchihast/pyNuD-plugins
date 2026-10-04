"""Data-dependent Particle Tracking analysis and validated result import.

Generated code runs only in the Codex execution sandbox. This module never execs
or imports model-authored Python into the GUI process. Original ASD stays outside
that writable workspace. Review and Apply remain separate operations.
"""
from ai_recovery import bounded_analysis, current_budget, is_timeout_message
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from background_auto import check_cancel
from codex_app_server import CodexError
from analysis_runtime import (agent_availability, create_agent_client, worker_command, digest,
                              read_workspace_file, result_schema, check_reply, review_measurements,
                              open_host_output, write_host_text, replay_timeout, isolated_replay, REPLAY_RULE)

FORMAT = 'pynud-particle-agent-v1'
MAX_RESULT_BYTES = 64 * 1024 * 1024

# Exported into the disposable workspace, including in frozen installations.
# Helpers also expose immutable measured checkpoints for targeted repairs.
SUPPORT_CODE = r'''import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def load():
    return np.load('frames.npy', mmap_mode='r', allow_pickle=False), json.loads(Path('metadata.json').read_text(encoding='utf-8'))

def load_checkpoint(index=0):
    return json.loads(Path(f'checkpoint_{int(index)}.json').read_text(encoding='utf-8'))

def save_checkpoint(index=0, replacements=None, algorithm=None, parameters=None, warnings=None):
    result=load_checkpoint(index)
    changes={int(k):v for k,v in (replacements or {}).items()}
    if not set(changes)<=set(r['frame'] for r in result['frames']):
        raise ValueError('Unknown replacement frame.')
    records=[changes.get(r['frame'],r) for r in result['frames']]
    save_result(records,algorithm or result['algorithm'],
        result['parameters'] if parameters is None else parameters,
        result['warnings'] if warnings is None else warnings)

def save_result(frames, algorithm, parameters=None, warnings=None):
    _, meta = load()
    result = dict(format='pynud-particle-agent-v1', input_sha256=meta['input_sha256'],
        algorithm=algorithm, parameters=parameters or {}, warnings=warnings or [], frames=frames)
    def scalar(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    Path('result.json').write_text(json.dumps(result, default=scalar, allow_nan=False), encoding='utf-8')

def preview(indices, output='agent_preview.png'):
    images,meta=load(); result=json.loads(Path('result.json').read_text(encoding='utf-8')); lut=np.load('palette.npy',allow_pickle=False)
    records={r['frame']:r for r in result['frames']}
    canvas=Image.new('RGB',(4*256, ((len(indices)+3)//4)*280),'#161a20');d=ImageDraw.Draw(canvas)
    for k,index in enumerate(indices):
        a=images[index];lo,hi=np.percentile(a,[1,99]);z=np.uint8(np.clip((a-lo)/max(hi-lo,1e-12),0,1)*255)
        tile=Image.fromarray(lut[np.flipud(z)])
        ratio=(a.shape[1]*meta['pixel_size_nm'][0])/(a.shape[0]*meta['pixel_size_nm'][1])
        w,h=(256, max(1,round(256/ratio))) if ratio>=1 else (max(1,round(256*ratio)),256)
        tile=tile.resize((w,h));draw=ImageDraw.Draw(tile);sx=w/a.shape[1];sy=h/a.shape[0]
        for p in records[index]['detections']:
            l,b,r,t=p['roi'];color='red' if p['review_required'] else '#40ffc0' if p['id'] is not None else 'yellow'
            draw.rectangle((l*sx,(a.shape[0]-t)*sy,min(w-1,r*sx),min(h-1,(a.shape[0]-b)*sy)),outline=color,width=2)
            draw.text((max(0,l*sx),max(0,(a.shape[0]-t)*sy-10)),str(p['id']),fill=color)
        x,y=k%4*256,k//4*280;canvas.paste(tile,(x,y+24));d.text((x+3,y+3),f"Frame {index+1} / {records[index]['status']}",fill='white')
    canvas.save(output)
    return str(Path(output).resolve())
'''


ARCHIVE_FORMAT = 'pynud-particle-analysis-v1'
MAX_INPUT_BYTES = 2 * 1024 ** 3


def export_analysis(result, path, original=None):
    """Explicit portable archive; retain the unfiltered review for reversible import."""
    import os
    import tempfile
    import zipfile
    from datetime import datetime, timezone
    from importlib.metadata import version, PackageNotFoundError
    path = Path(path)
    if path.suffix.lower() != '.zip': raise ValueError('Choose a .zip analysis archive.')
    result['source'].assert_unchanged()
    record = result['agent_record']
    frames = result['frames']
    original = result if original is None else original
    if original is result and result.get('review_exclusions', {}).get('observations', 0):
        raise ValueError('Export requires the original review result to retain excluded observations.')
    versions = {}
    for package in ('numpy', 'scipy', 'scikit-image', 'opencv-python', 'Pillow', 'pandas'):
        try: versions[package] = version(package)
        except PackageNotFoundError: pass
    session = dict(format=ARCHIVE_FORMAT, saved_at=datetime.now(timezone.utc).isoformat(),
        source_name=result['source'].path.name, source_sha256=digest(result['source'].path),
        settings=result['settings'], scope=result['scope'], reason=result['reason'],
        instructions=result.get('instructions', ''), ai_provider=result.get('ai_provider'),
        ai_model=result.get('ai_model'), python_version=sys.version, library_versions=versions)
    fd, temporary = tempfile.mkstemp(prefix='.pynud-analysis-', suffix='.zip', dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            def write_json(name, value): archive.writestr(name, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
            write_json('session.json', session)
            archive.writestr('analysis.py', record['script'])
            archive.writestr('analysis_support.py', record.get('support_script', SUPPORT_CODE))
            write_json('metadata.json', record['input_metadata'])
            write_json('result.json', record['result'])
            checkpoints=record.get('checkpoints',{})
            for index,raw in checkpoints.items():
                write_json(f'checkpoint_{int(index)}.json',raw)
            if record.get('refinement_input') is not None:
                write_json('previous.json', record['refinement_input'])
            if record.get('previous_analysis_script'):
                archive.writestr('previous_analysis.py', record['previous_analysis_script'])
            write_json('review.json', dict(reports=record['reports'],
                ai_review_status=record.get('ai_review_status'),
                checkpoint_indices=[int(i) for i in checkpoints],
                script_sha256=record['script_sha256'], warnings=result['warnings'],
                review_exclusions=result.get('review_exclusions', {}),
                unfiltered_result=export_previous(original), final_selection=export_previous(result),
                viewed_by_host_review_frames_0based=record['viewed_by_host_review_frames_0based']))
            archive.writestr('selected_tracks.csv', result['tracks'].to_csv(index=False))
            with archive.open('frames.npy', 'w', force_zip64=True) as target:
                np.lib.format.write_array_header_1_0(target, dict(descr=np.dtype('float64').str,
                    fortran_order=False, shape=(len(frames),)+frames[0].shape))
                for image in frames: target.write(np.asarray(image, dtype='float64').tobytes(order='C'))
            with archive.open('palette.npy', 'w') as target:
                palette=result['palette']
                if palette is None: palette=np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,axis=1)
                np.save(target, np.asarray(palette, dtype=np.uint8), allow_pickle=False)
        result['source'].assert_unchanged()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def archive_has_source_identity(path):
    import zipfile
    with zipfile.ZipFile(path) as archive:
        return 'session.json' in archive.namelist()


def _archive_text(archive, name, limit=MAX_RESULT_BYTES):
    info = archive.getinfo(name)
    if info.file_size > limit: raise ValueError('Oversized analysis archive entry: ' + name)
    return archive.read(name).decode('utf-8')


def _archive_json(archive, name):
    def invalid(value): raise ValueError('Non-finite JSON value: ' + value)
    data = json.loads(_archive_text(archive, name), parse_constant=invalid)
    if not isinstance(data, dict): raise ValueError('Invalid archive object: ' + name)
    return data


def _archive_array(archive, name, shape, dtype, cancelled):
    """Validate the NPY header before allocating; never unpickle or extract files."""
    import math
    size = math.prod(shape) * np.dtype(dtype).itemsize
    if size > MAX_INPUT_BYTES: raise ValueError('The analysis input exceeds the 2 GiB import limit.')
    with archive.open(name) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0): actual, order, kind = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0): actual, order, kind = np.lib.format.read_array_header_2_0(stream)
        else: raise ValueError('Unsupported NPY format: ' + name)
        if actual != tuple(shape) or order or kind != np.dtype(dtype):
            raise ValueError('Invalid array dimensions or dtype: ' + name)
        if archive.getinfo(name).file_size != stream.tell() + size:
            raise ValueError('Truncated or oversized array: ' + name)
        array = np.empty(shape, dtype=dtype)
        target = memoryview(array).cast('B')
        for offset in range(0, size, 1024 * 1024):
            check_cancel(cancelled)
            block = stream.read(min(1024 * 1024, size-offset))
            if len(block) != min(1024 * 1024, size-offset): raise ValueError('Truncated array: ' + name)
            target[offset:offset+len(block)] = block
    return array


def _checked_display(display):
    if display is None: return None  # Older archives use the original percentile preview.
    if not isinstance(display, dict) or display.get('contrastMode') not in ('Auto1','Auto2','Auto3','Manual','LAFM'):
        raise ValueError('Invalid saved display settings.')
    for key in ('autoContrastMin','autoContrastMax','manualContrastMin','manualContrastMax'):
        value=display.get(key)
        if value is not None and (type(value) not in (float,int) or not np.isfinite(value)):
            raise ValueError('Invalid saved contrast range.')
    if 'gamma_lut' in display:
        tone=display['gamma_lut']
        if not isinstance(tone,list) or len(tone)!=256 or any(type(v) is not int or not 0<=v<=255 for v in tone):
            raise ValueError('Invalid saved gamma LUT.')
    if display.get('clahe'):
        clahe=display['clahe']; grid=clahe.get('tile_grid_size') if isinstance(clahe,dict) else None
        clip=clahe.get('clip_limit') if isinstance(clahe,dict) else None
        if (not isinstance(grid,list) or len(grid)!=2 or any(type(v) is not int or not 1<=v<=256 for v in grid)
                or type(clip) not in (float,int) or not np.isfinite(clip) or not 0<clip<=100):
            raise ValueError('Invalid saved CLAHE settings.')
    return display


def import_analysis(path, source, fallback_settings=None, cancelled=lambda:False, progress=lambda p,m:None):
    """Restore measurements locally. No imported Python is executed; no AI client is used.

    For legacy ZIPs without source identity, source must replay the exact saved
    numerical input (the UI captures current main processing for this case).
    """
    import zipfile
    from dataclasses import replace
    from background_auto import ASDSource
    from .ParticleTracking import ai_settings_checked, ai_filter_result, ai_measure_result, ai_preview_image
    source.assert_unchanged()
    try:
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)): raise ValueError('Duplicate entries in the analysis archive.')
            progress(5, 'Checking saved input and source ASD…'); check_cancel(cancelled)
            session=_archive_json(archive,'session.json') if 'session.json' in names else None
            if session is not None:
                if session.get('format')!=ARCHIVE_FORMAT: raise ValueError('Unsupported analysis archive version.')
                if session.get('source_sha256')!=digest(source.path):
                    raise ValueError('This analysis belongs to a different ASD. Open its original source file first.')
            metadata=_archive_json(archive,'metadata.json')
            shape=metadata.get('shape'); px=metadata.get('pixel_size_nm')
            if (metadata.get('format')!=FORMAT or not isinstance(shape,list) or len(shape)!=3
                    or any(type(v) is not int or v<1 for v in shape) or min(shape[1:])<3
                    or shape[0]!=source.header.frame_num or metadata.get('height_unit')!='nm'
                    or not isinstance(px,list) or len(px)!=2
                    or any(type(v) not in (float,int) or not np.isfinite(v) or v<=0 for v in px)):
                raise ValueError('Invalid saved frame dimensions or calibration.')
            movie=_archive_array(archive,'frames.npy',shape,'float64',cancelled)
            if not np.isfinite(movie).all(): raise ValueError('Non-finite saved image data.')
            checksum=hashlib.sha256()
            with archive.open('frames.npy') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    check_cancel(cancelled); checksum.update(block)
            if checksum.hexdigest()!=metadata.get('input_sha256'):
                raise ValueError('Saved image data failed the SHA-256 integrity check.')
            if session is None:
                current_px=[source.header.x_scan_size/source.shape[1],source.header.y_scan_size/source.shape[0]]
                if source.shape!=tuple(shape[1:]) or not np.allclose(px,current_px,rtol=1e-12,atol=0):
                    raise ValueError('Legacy archive calibration differs from the current main-window input.')
                for i,image in enumerate(movie):
                    check_cancel(cancelled)
                    if not np.array_equal(image,source.read(i)):
                        raise ValueError('Legacy archive has no source identity. Restore the original background/filter settings '
                                         'in the main window before importing; its numerical input must match exactly.')
            palette=_archive_array(archive,'palette.npy',(256,3),'uint8',cancelled)
            raw=_archive_json(archive,'result.json'); review=_archive_json(archive,'review.json')
            script=_archive_text(archive,'analysis.py',1024*1024)
            if hashlib.sha256(script.encode('utf-8')).hexdigest()!=review.get('script_sha256'):
                raise ValueError('Saved analysis code failed the SHA-256 integrity check.')
            display=_checked_display(metadata.get('display_settings'))
            frame_time=metadata.get('frame_time_ms',source.header.frame_time)
            if frame_time is None: frame_time=source.header.frame_time
            if type(frame_time) not in (int,float) or not np.isfinite(frame_time) or frame_time<=0:
                raise ValueError('Invalid saved frame time.')
            progress(50, 'Validating particle ROIs, IDs and review choices…'); check_cancel(cancelled)
            validate_result(raw,movie,metadata['input_sha256'])
            indices=review.get('checkpoint_indices',[])
            if (not isinstance(indices,list) or len(indices)>100
                    or any(type(i) is not int or not 0<=i<100 for i in indices) or len(set(indices))!=len(indices)):
                raise ValueError('Invalid saved checkpoint indices.')
            checkpoints={}
            for index in indices:
                cached=_archive_json(archive,f'checkpoint_{index}.json')
                validate_result(cached,movie,metadata['input_sha256'])
                checkpoints[str(index)]=cached
            base=review.get('unfiltered_result',dict(frames=raw['frames']))
            if not isinstance(base,dict): raise ValueError('Invalid saved unfiltered review.')
            detected,tracks,states=validate_result(dict(raw,frames=base.get('frames')),movie,metadata['input_sha256'])
            exclusions=review.get('review_exclusions',{})
            if not isinstance(exclusions,dict) or any(type(exclusions.get(k,False)) is not bool for k in ('red','yellow')):
                raise ValueError('Invalid saved exclusion choices.')
            warnings=review.get('warnings',[])
            if not isinstance(warnings,list) or any(not isinstance(w,str) for w in warnings): raise ValueError('Invalid saved warnings.')
            saved_settings=ai_settings_checked(session['settings'] if session else fallback_settings)
            scope=session.get('scope') if session else [1,len(movie)]
            if not isinstance(scope,list) or len(scope)!=2 or any(type(v) is not int for v in scope) or not 1<=scope[0]<=scope[1]<=len(movie):
                raise ValueError('Invalid saved refinement range.')
            reason=session.get('reason') if session else raw['algorithm']
            if not isinstance(reason,str): raise ValueError('Invalid saved analysis explanation.')
            shown=review.get('viewed_by_host_review_frames_0based',[])
            if not isinstance(shown,list) or any(type(f) is not int or not 0<=f<len(movie) for f in shown):
                raise ValueError('Invalid saved review frame list.')
            # Bind saved numerical frames to this verified ASD, independent of
            # current background/filter settings. It is used only by tracking.
            class SavedInputSource(ASDSource):
                def read(self, index):
                    self.assert_unchanged()
                    if not 0<=index<len(self.images): raise IndexError(index)
                    return self.images[index]
            restored=SavedInputSource(source.path)
            restored.images=movie; restored.shape=tuple(shape[1:])
            restored.header=replace(restored.header,x_pixel=shape[2],y_pixel=shape[1],
                x_scan_size=px[0]*shape[2],y_scan_size=px[1]*shape[1],frame_time=frame_time)
            result=dict(engine='agent',source=restored,frames=movie,palette=palette,display_settings=display,
                smoothing=('None',1.),settings=saved_settings,scope=scope,reason=reason,warnings=warnings,
                detections=detected,tracks=tracks,frame_assessments=states,metrics=ai_measure_result(detected,tracks,px),
                review_exclusions=exclusions,imported_archive=str(Path(path).resolve()),
                instructions=session.get('instructions','') if session else '',
                ai_provider=session.get('ai_provider') if session else None,ai_model=session.get('ai_model') if session else None,
                agent_record=dict(input_metadata=metadata,result=raw,script=script,script_sha256=review['script_sha256'],
                    checkpoints=checkpoints,
                    support_script=_archive_text(archive,'analysis_support.py',1024*1024),reports=review.get('reports',[]),
                    refinement_input=_archive_json(archive,'previous.json') if 'previous.json' in names else None,
                    previous_analysis_script=_archive_text(archive,'previous_analysis.py',1024*1024) if 'previous_analysis.py' in names else None,
                    viewed_by_host_review_frames_0based=shown,protocol_events=[],
                    ai_review_status=review.get('ai_review_status')))
            if not isinstance(result['instructions'],str): raise ValueError('Invalid saved instructions.')
            selected=ai_filter_result(result,exclusions.get('red',False),exclusions.get('yellow',False))
            if export_previous(selected)!=review.get('final_selection'):
                raise ValueError('Saved review choices do not match the selected result. Export the analysis again.')
            # Check display restoration before the GUI replaces an existing result.
            ai_preview_image(movie[0],detected[0],tracks[tracks.frame==0],palette,px,display=display)
            source.assert_unchanged(); check_cancel(cancelled)
            movie.flags.writeable=False
            progress(100, 'Analysis imported locally. Review, then Apply to Particle Tracking. No AI request was made.')
            return result
    except (KeyError, TypeError, UnicodeError, zipfile.BadZipFile, EOFError, OverflowError) as exc:
        raise ValueError('Invalid or incomplete analysis archive: '+str(exc)) from exc


def validate_result(raw, images, input_hash):
    """Never accept missing frames, out-of-image geometry or duplicate identities."""
    from .ParticleTracking import Particle
    import pandas as pd
    if (not isinstance(raw, dict) or raw.get('format') != FORMAT or raw.get('input_sha256') != input_hash
            or not isinstance(raw.get('algorithm'), str) or not raw['algorithm'].strip()
            or not isinstance(raw.get('parameters'), dict) or not isinstance(raw.get('warnings'), list)
            or any(not isinstance(w, str) for w in raw['warnings'])):
        raise ValueError('Invalid analysis report or mismatched input data.')
    records = raw.get('frames')
    if not isinstance(records, list) or len(records) != len(images):
        raise ValueError('The result must assess every frame, including frames with no particles.')
    height, width = images[0].shape
    detected, rows, states, seen, total = {}, [], [], set(), 0
    for record in records:
        if not isinstance(record, dict): raise ValueError('Invalid frame record.')
        f = record.get('frame')
        if type(f) is not int or not 0 <= f < len(images) or f in seen:
            raise ValueError('Duplicate or out-of-range frame number.')
        seen.add(f)
        if record.get('status') not in ('ok', 'uncertain', 'unusable') or not isinstance(record.get('reason'), str):
            raise ValueError(f'Frame {f+1}: missing assessment.')
        particles = record.get('detections')
        if not isinstance(particles, list) or len(particles) > min(10000, height * width):
            raise ValueError('Invalid or excessive particle detections.')
        total += len(particles)
        if total > 1000000: raise ValueError('The result exceeds one million observations.')
        if record['status'] == 'unusable' and (particles or not record['reason'].strip()):
            raise ValueError('Unusable frames need a reason and must not contain invented observations.')
        detected[f], ids = [], set()
        for p in particles:
            if not isinstance(p, dict) or 'id' not in p: raise ValueError('Invalid particle observation or missing ID.')
            coords = [p.get('x'), p.get('y')]
            roi = p.get('roi')
            if (not isinstance(roi, list) or len(roi) != 4 or
                    any(type(v) not in (int, float) or not np.isfinite(v) for v in coords + roi)):
                raise ValueError('Non-finite or missing particle coordinates/ROI.')
            x, y = map(float, coords); left, bottom, right, top = roi
            if not (0 <= x < width and 0 <= y < height and 0 <= left < right <= width
                    and 0 <= bottom < top <= height and left <= x+.5 <= right and bottom <= y+.5 <= top):
                raise ValueError(f'Frame {f+1}: ROI must enclose the measured center within the image.')
            ident = p.get('id')
            if ident is not None and (type(ident) is not int or not 0 <= ident <= 1000000000 or ident in ids):
                raise ValueError(f'Frame {f+1}: invalid or duplicate track ID.')
            if type(p.get('review_required')) is not bool:
                raise ValueError('Every observation must have an explicit review_required flag.')
            if ident is not None: ids.add(ident)
            flag = p['review_required'] or record['status'] == 'uncertain'
            intensity = float(images[f][min(height-1,int(round(y))), min(width-1,int(round(x)))])
            particle = Particle(f, x, y, intensity, max(1., min(right-left, top-bottom)/2), 1., ident,
                                tuple(map(float, roi)), flag)
            detected[f].append(particle)
            if ident is not None:
                rows.append(dict(frame=f, x=x, y=y, particle=ident, review_required=flag, match_similarity=None))
        states.append(dict(frame=f, status=record['status'], reason=record['reason']))
    tracks = pd.DataFrame(rows, columns=['frame','x','y','particle','review_required','match_similarity'])
    tracks = tracks.sort_values(['frame', 'particle']).reset_index(drop=True)
    validate_coordinate_alignment(detected, images)
    return detected, tracks, sorted(states,key=lambda r:r['frame'])


def validate_coordinate_alignment(detected, images):
    """Catch a strong systematic y-reflection error before ROI import.

    This is a diagnostic, not a particle classifier. Both bright and dark
    extrema count, with tied background pixels at their mid-rank. Never flip
    coordinates automatically: ask the agent to check its measured geometry.
    """
    direct, reflected = [], []
    for frame, particles in detected.items():
        if not particles: continue
        image = images[frame]; ordered = np.sort(image.ravel())
        for p in particles:
            x = min(image.shape[1]-1, int(round(p.x)))
            y = min(image.shape[0]-1, int(round(p.y)))
            values = [image[y,x], image[image.shape[0]-1-y,x]]
            ranks = (np.searchsorted(ordered,values,'left') + np.searchsorted(ordered,values,'right')) / (2*len(ordered))
            direct.append(abs(ranks[0]-.5)*2); reflected.append(abs(ranks[1]-.5)*2)
    if len(direct) < 8: return
    difference = np.asarray(reflected) - direct
    if np.median(difference) > .4 and np.mean(difference > 0) >= .9:
        raise ValueError('Possible vertically reflected coordinates: the mirrored positions align with image extrema '
            'substantially better than the reported centers. Check array indexing against the supplied images. '
            'Report x=column and y=NUMPY ROW INDEX directly; do not convert y to height-1-row. '
            'The pyNuD renderer already flips image and overlays together. ROI edges must use the same array-row coordinates.')


def frame_metrics(images, cancelled):
    result=[]
    for i,a in enumerate(images):
        check_cancel(cancelled)
        levels=np.percentile(a,[1,50,99])
        row=np.median(a,axis=1)
        result.append(dict(frame=i, height_p01=float(levels[0]), height_median=float(levels[1]), height_p99=float(levels[2]),
                           max_row_step=float(np.max(np.abs(np.diff(row)))), standard_deviation=float(np.std(a))))
    return result


def review_indices(count, detected=None, states=None, limit=128):
    """Even coverage, count changes and flagged frames spread over the whole movie."""
    if count <= 96: return list(range(count))
    selected=set(np.linspace(0,count-1,32,dtype=int).tolist())
    if detected:
        counts=np.array([len(detected[i]) for i in range(count)])
        for i in np.argsort(-np.abs(np.diff(counts)))[:16]: selected.update([int(i),int(i)+1])
    flagged=sorted({int(s['frame']) for s in states or () if s['status']!='ok'})
    if flagged:
        room=max(1,(limit-len(selected))//3)
        for f in np.asarray(flagged)[np.linspace(0,len(flagged)-1,min(len(flagged),room),dtype=int)]:
            selected.update(range(max(0,int(f)-1),min(count,int(f)+2)))
    ordered=sorted(selected)
    if len(ordered)>limit: ordered=[ordered[i] for i in np.linspace(0,len(ordered)-1,limit,dtype=int)]
    return ordered


def panels(images, detected, tracks, indices, root, prefix, palette, pixel_size, display=None):
    from .ParticleTracking import ai_preview_image
    from PIL import Image, ImageDraw, ImageOps
    paths=[]
    for start in range(0,len(indices),16):
        chosen=indices[start:start+16]
        canvas=Image.new('RGB',(1024,((len(chosen)+3)//4)*280),'#171b21'); draw=ImageDraw.Draw(canvas)
        for j,index in enumerate(chosen):
            selected=tracks[tracks.frame==index] if not tracks.empty else tracks
            tile=ai_preview_image(images[index],detected[index],selected,palette,pixel_size,display=display)
            tile=ImageOps.contain(tile,(252,252))
            x,y=j%4*256,j//4*280
            canvas.paste(tile,(x,y+24));draw.text((x+3,y+4),f'Frame {index+1}: {len(detected[index])} detections',fill='white')
        name=f'{prefix}_{start}.png'
        with open_host_output(root,name) as stream: canvas.save(stream,format='PNG')
        paths.append(str(Path(root)/name))
    return paths


def export_previous(previous):
    states={r['frame']:r for r in previous.get('frame_assessments',[])}
    records=[]
    for f,particles in sorted(previous['detections'].items()):
        flagged=set(previous['tracks'].loc[(previous['tracks'].frame==f)&previous['tracks'].review_required,'particle'])
        state=states.get(f,dict(status='ok',reason='Initial local detection; not yet reviewed by AI or a person.'))
        records.append(dict(frame=f,status=state['status'],reason=state['reason'],detections=[dict(x=p.x,y=p.y,
            roi=list(p.roi or (max(0,p.x+.5-p.radius),max(0,p.y+.5-p.radius),
                min(previous['frames'][f].shape[1],p.x+.5+p.radius),min(previous['frames'][f].shape[0],p.y+.5+p.radius))),
            id=p.track_id,review_required=bool(p.track_id in flagged or p.size_unresolved)) for p in particles]))
    return dict(frames=records)


def validate_refine_scope(raw, old, scope):
    if old is None: return
    fresh={r['frame']:r for r in raw['frames']}
    for record in old['frames']:
        f=record['frame']
        if scope[0] <= f <= scope[1]: continue
        # Refinement can relink identities globally, but must preserve measured
        # observations and acquisition assessments outside the requested range.
        def observations(r):
            return sorted((p['x'],p['y'],tuple(p['roi'])) for p in r['detections'])
        if (observations(record)!=observations(fresh[f]) or record['status']!=fresh[f]['status']):
            raise ValueError(f'Refine changed observations outside the requested range (frame {f+1}).')


@bounded_analysis
def run_particle_agent(images, settings, pixel_size, client, workspace, instructions='', palette=None,
                       cancelled=lambda:False, progress=lambda p,m:None, previous=None, scope=None,
                       frame_time_ms=None, max_rounds=2, time_limit=900, display=None):
    """Reuse measured observations; allow one bounded correction after AI review."""
    from .ParticleTracking import ai_measure_result, ai_detect_frame, ai_link_particles
    import pandas as pd
    if (not len(images) or any(a.ndim!=2 or a.shape!=images[0].shape or min(a.shape)<3 or not np.isfinite(a).all() for a in images)):
        raise ValueError('Agent input requires finite, equally sized numerical frames.')
    if len(pixel_size)!=2 or not all(np.isfinite(p) and p>0 for p in pixel_size): raise ValueError('Invalid pixel sizes.')
    first,last=scope if scope is not None else (0,len(images)-1)
    if not 0<=first<=last<len(images) or (scope is not None and previous is None): raise ValueError('Invalid Refine range.')
    if type(max_rounds) is not int or not 1<=max_rounds<=12: raise ValueError('AI rounds must be between 1 and 12.')
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    budget = current_budget()
    def check(check_time=True):
        check_cancel(cancelled)
        budget.remaining(check_time)
    def report(percent,message): check();progress(percent,message)
    report(2,'Preparing all numerical frames for the analysis agent…')
    array=np.lib.format.open_memmap(root/'frames.npy',mode='w+',dtype='float64',shape=(len(images),)+images[0].shape)
    for i,a in enumerate(images): check();array[i]=a;budget.touch()
    array.flush();del array
    input_hash=digest(root/'frames.npy')
    palette=np.asarray(palette,dtype=np.uint8).reshape(256,3) if palette is not None else np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,axis=1)
    np.save(root/'palette.npy',palette,allow_pickle=False)
    metadata=dict(format=FORMAT,input_sha256=input_hash,shape=[len(images),*images[0].shape],height_unit='nm',
        pixel_size_nm=list(pixel_size),frame_time_ms=frame_time_ms,refine_range_0based=[first,last],display_settings=display,
        data_description='Main-window processed height images captured at Start, without additional plugin smoothing. Do not double-apply background corrections.',
        coordinates='frame is zero-based. x=column and y=NUMPY ROW INDEX directly. NEVER convert row to height-1-row. '
            'The renderer already flips the input image and overlays together. Example: a detection at array[10,20] '
            'must be x=20,y=10, irrespective of how an image viewer displays it. '
            'ROI uses ARRAY pixel edges [column_min,row_min,column_max_exclusive,row_max_exclusive]; center is [x+0.5,y+0.5].',
        frame_metrics=frame_metrics(images,cancelled))
    (root/'metadata.json').write_text(json.dumps(metadata,allow_nan=False),encoding='utf-8')
    (root/'analysis_support.py').write_text(SUPPORT_CODE,encoding='utf-8')
    old=export_previous(previous) if previous is not None else None
    if old: (root/'previous.json').write_text(json.dumps(old,allow_nan=False),encoding='utf-8')
    immutable=['frames.npy','metadata.json','palette.npy','analysis_support.py']+(['previous.json'] if old else [])
    prior_script = (previous.get('agent_record', {}).get('script') if previous else None)
    if prior_script:
        (root/'previous_analysis.py').write_bytes(prior_script.encode('utf-8'))
        immutable.append('previous_analysis.py')
    original_hashes={name:digest(root/name) for name in immutable}
    def verify_inputs(check_time=True):
        check(check_time)
        if any((root/name).is_symlink() or digest(root/name)!=value for name,value in original_hashes.items()):
            raise ValueError('The agent changed its input snapshot. No result was imported.')
    checkpoints={}
    def checkpoint(index, raw):
        name=f'checkpoint_{index}.json'
        write_host_text(root,name,json.dumps(raw,allow_nan=False))
        original_hashes[name]=digest(root/name)
        checkpoints[str(index)]=deepcopy(raw)
        return name
    if old is None:
        report(3,'Measuring initial particle detections once; AI will repair only demonstrated problems…')
        detections={}
        for f,image in enumerate(images):
            check();detections[f]=ai_detect_frame(image,settings,pixel_size,f)
            report(3+int(3*(f+1)/len(images)),f'Initial detection: frame {f+1}/{len(images)}')
        detections,tracks=ai_link_particles(images,detections,settings,pixel_size,cancelled)
        initial=export_previous(dict(frames=images,detections=detections,tracks=tracks))
        algorithm='Initial local detection and ID linking; pending AI review.'
    else:
        initial=old
        algorithm='Previously measured observations and IDs; pending range refinement.'
    initial=dict(initial,format=FORMAT,input_sha256=input_hash,algorithm=algorithm,
                 parameters=deepcopy(settings),warnings=[])
    initial_detected,initial_tracks,initial_states=validate_result(initial,images,input_hash)
    initial_name=checkpoint(0,initial)
    immutable.append(initial_name)
    (root/'analysis.py').write_text('from analysis_support import save_checkpoint\nsave_checkpoint(0)\n',encoding='utf-8')
    overview=panels(images,{i:[] for i in range(len(images))},pd.DataFrame(),
        np.linspace(0,len(images)-1,min(16,len(images)),dtype=int).tolist(),root,'input',palette,pixel_size,display)
    overview+=panels(images,initial_detected,initial_tracks,review_indices(len(images),initial_detected,initial_states),
                     root,'initial',palette,pixel_size,display)
    command=worker_command(root)
    contract=dict(format=FORMAT,input_sha256=input_hash,algorithm='Explain actual chosen algorithms and why they fit this data.',
        parameters={'example':'record actual numerical parameters; any algorithm is allowed'},warnings=['Limitations, if any.'],
        frames=[dict(frame=0,status='ok',reason='Assessment of this frame.',
                     detections=[dict(x=10.,y=10.,roi=[5.,5.,16.,16.],id=1,review_required=False)])])
    payload=dict(task='Analyze this AFM movie as a data-dependent particle detection and tracking task. '
        'Start from checkpoint_0.json and the supplied analysis.py, which reuse measured detections and IDs. '
        'Inspect the initial ROIs; retain correct observations and repair concrete defects or requested ranges. '
        'If the initial detector systematically fails on this data, replace it with a justified numerical method. '
        'Do not run obligatory parameter sweeps or rebuild good results. Make one focused correction pass, '
        'execute analysis.py, inspect targeted overlays, then deliver result.json with unresolved issues flagged. '
        'Do not merely return detector settings. The example contract is a schema illustration, never data to copy.',
        workspace_files=immutable,
        runtime_command=command,available_libraries=['numpy','scipy','scikit-image','cv2','PIL','pandas'],
        helpers='from analysis_support import load, save_result, preview, load_checkpoint, save_checkpoint; load() returns frames,metadata; '
            'load_checkpoint(index) returns measured frames. save_checkpoint(index,replacements={frame:record}) '
            'copies all unchanged frames and merges repaired records. For ID-only repairs, reuse coordinates/ROIs '
            'and change only the IDs/flags in these measured records. '
            'save_result(frame_records,algorithm,parameters,warnings) writes result.json; preview([zero-based frames]) renders ROI panels.',
        result_contract=contract,user_instructions=instructions,
        coordinates=metadata['coordinates'],
        requirements=[
            'Inspect all-frame numerical metrics and representative images, then request/view additional consecutive frames wherever needed. '
            'You can write any numerical algorithm using the available libraries. Adapt to THIS data; no fixed unusable frames or sample-specific rules.',
            'Measure centers and ROIs from numerical pixels. Never type guessed coordinates or impose a desired particle count. '
            'Inspect missed dim particles, false scan-line detections, multiple peaks within one object, close particles, births/disappearances and merged objects.',
            'Consider local-background-relative segmentation, noise-adaptive thresholds, LoG or other methods as evidence warrants. '
            'Do not assume every bright peak is a separate molecule. Preserve geometry and physical calibration.',
            'Consider common field motion separately from particle motion. Only use it when supported by independent image/particle evidence. '
            'Keep displayed coordinates in the original input frame. Do not crop or move image pixels.',
            'Analyze every frame. Mark acquisition failures unusable with an explanation and zero detections. '
            'Never interpolate unobserved coordinates or force identity through unobservable gaps. Flag identity ambiguity.',
            'Every frame record must contain frame,status,reason,detections. Every detection needs x,y,roi,id,review_required. '
            'Status: ok, uncertain or unusable. IDs are nonnegative integers unique within each frame, or null for unlinked detections. '
            'Use id=null for isolated detections without reliable temporal support, and report the retention criterion chosen for this data. '
            'Use ROIs enclosing the full measured object with small margins; overlaps/mergers require review flags.',
            'Write only within this workspace. Preserve all supplied input/helper files. No installs, internet or unrelated file access. '
            'Use the supplied runtime command to execute analysis.py; it must deterministically regenerate result.json from the supplied data. Do not leave background commands running. '+REPLAY_RULE,
            'Render and inspect measured overlays before finishing. Record limitations honestly. '
            'No synthetic coordinates, no manual lists of detections, no claiming visual review of unseen frames.'])
    if old:
        payload['requirements'].append(f'Refine only observations in frames {first+1}–{last+1} (1-based). '
            'Use previous.json to keep measured coordinates/ROIs and status outside that range exactly. '
            'When previous_analysis.py is supplied, inspect it as the previous method and improve analysis.py based on the new evidence. '
            'You may relink IDs globally to maintain continuity. Do not discard earlier corrections outside the range.')
    reports=[]; last_error=''; audit_paths=[]
    last_result=None
    seed_script='from analysis_support import save_checkpoint\nsave_checkpoint(0)\n'
    if old is None:
        retained_result=dict(engine='agent',settings=deepcopy(settings),detections=initial_detected,tracks=initial_tracks,
            metrics=ai_measure_result(initial_detected,initial_tracks,pixel_size),frame_assessments=initial_states,
            scope=[first+1,last+1],reason=algorithm,warnings=[],
            agent_record=dict(input_metadata=metadata,result=deepcopy(initial),script=seed_script,reports=[],
                              script_sha256=hashlib.sha256(seed_script.encode('utf-8')).hexdigest(),
                              viewed_by_host_review_frames_0based=[],protocol_events=[],
                              support_script=SUPPORT_CODE,checkpoints=deepcopy(checkpoints),ai_review_status='pending'))
    else:
        # A Refine that cannot finish leaves the reviewed result it started from
        # in place: every per-observation flag, decision and script survives.
        retained_result={k:deepcopy(v) for k,v in previous.items() if k!='frames'}
        retained_result.setdefault('warnings',[]); retained_result.setdefault('agent_record',{})
    def flag_all_for_review(result):
        result['tracks']['review_required']=True
        for group in result['detections'].values():
            for particle in group: particle.size_unresolved=True
        result['metrics']=ai_measure_result(result['detections'],result['tracks'],pixel_size)
    def retain_measured(message):
        verify_inputs(check_time=False)
        result=deepcopy(retained_result)
        result['warnings'].append(message)
        result['reason']=str(result.get('reason',''))+'\n'+message
        result['agent_record']['ai_review_status']='timed_out'
        # Only an initial run has nothing AI-reviewed yet. A Refine keeps the
        # existing per-observation triage so remaining problems stay localized.
        if old is None: flag_all_for_review(result)
        return result
    budget.checkpoint=retain_measured
    base_payload=deepcopy(payload)
    for attempt in range(max_rounds):
        check();client.settings['timeout']=max(1,int(budget.remaining()))
        percent=10+attempt*25
        report(percent,f'AI analyzing the data and testing its code — round {attempt+1}/{max_rounds}…')
        answer=check_reply(client.run_agent(payload,overview,result_schema(),lambda text:report(percent,text)))
        reports.append(answer)
        verify_inputs()
        try:
            script=read_workspace_file(root,'analysis.py',1024*1024)
            if not script.strip(): raise ValueError('The analysis script is empty.')
            output=root/'result.json'
            output.unlink(missing_ok=True)
            submitted=digest(root/'analysis.py')
            with isolated_replay(root,original_hashes,discard=('result.json',)):
                client.execute_analysis(command,timeout=replay_timeout(client,(len(images),)+tuple(images[0].shape)))
            verify_inputs()
            if (root/'analysis.py').is_symlink() or digest(root/'analysis.py')!=submitted:
                raise ValueError('The analysis script changed during verification.')
            raw=json.loads(read_workspace_file(root,'result.json',MAX_RESULT_BYTES),
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON value: '+value)))
            detected,tracks,states=validate_result(raw,images,input_hash)
            validate_refine_scope(raw,old,(first,last))
        except (ValueError, OSError, CodexError) as exc:
            # A stopped/expired sandbox run is a timeout, never a defect for another AI round.
            if isinstance(exc, TimeoutError) or is_timeout_message(exc): raise
            check();verify_inputs()
            last_error=str(exc)
            report(percent+5,'Result validation found a problem; asking AI to repair it: '+last_error[:500])
            payload={**base_payload,'task':'Repair only the validation failure. Reuse the latest immutable checkpoint '
                'for correct observations; regenerate the full result through save_checkpoint. Do not guess coordinates.',
                'validation_error':last_error,'latest_checkpoint':max(map(int,checkpoints))}
            continue
        metric=ai_measure_result(detected,tracks,pixel_size)
        checkpoint(attempt+1,raw)
        # Preserve the newest validated measurements before any more AI work.
        retained_result=dict(engine='agent',settings=deepcopy(settings),detections=detected,tracks=tracks,metrics=metric,
            frame_assessments=states,scope=[first+1,last+1],reason=raw['algorithm'],warnings=list(raw['warnings']),
            agent_record=dict(input_metadata=metadata,result=deepcopy(raw),script=script,reports=deepcopy(reports),
                support_script=SUPPORT_CODE,checkpoints=deepcopy(checkpoints),refinement_input=old,
                previous_analysis_script=prior_script,script_sha256=digest(root/'analysis.py'),
                protocol_events=list(client.events),viewed_by_host_review_frames_0based=[]))
        shown=review_indices(len(images),detected,states)
        audit_paths=panels(images,detected,tracks,shown,root,f'review{attempt}',palette,pixel_size,display)
        before=digest(root/'result.json');code_hash=digest(root/'analysis.py')
        report(percent+12,f'AI inspecting measured ROI results on {len(shown)} frames; all {len(images)} frames checked numerically…')
        client.settings['timeout']=max(1,int(budget.remaining()))
        client.progress=lambda text:report(percent+12,text)
        review=review_measurements(lambda:client.advise(dict(task='Review these pyNuD-rendered overlays and measured results. '
            'Do the ROIs cover the visible particles? Are IDs consistent, and scan failures/mergers flagged? '
            'Accept if ready for user review; otherwise name specific frames, IDs and defects needing repair. '
            'Use only supplied evidence. Do not execute code or repeat numerical analysis. '
            'Do not claim visual inspection of unsupplied frames.',
            metrics=metric,algorithm=raw['algorithm'],frame_assessments=states,shown_frames_0based=shown,
            user_instructions=instructions),audit_paths,result_schema(True)),verify_inputs)
        reports.append(review);verify_inputs()
        if digest(root/'result.json')!=before or digest(root/'analysis.py')!=code_hash:
            raise ValueError('Analysis changed during read-only review. No result was imported.')
        last_result=dict(engine='agent',settings=deepcopy(settings),detections=detected,tracks=tracks,metrics=metric,
            frame_assessments=states,scope=[first+1,last+1],reason=raw['algorithm']+'\n\n'+review['summary'],
            # Old rejected-review warnings stay in the audit history. Do not
            # present already repaired defects as warnings on the final result.
            warnings=list(dict.fromkeys(raw['warnings']+answer['warnings']+review['warnings'])),
            agent_record=dict(input_metadata=metadata,result=raw,script=script,reports=deepcopy(reports),
                support_script=SUPPORT_CODE,checkpoints=deepcopy(checkpoints),
                refinement_input=old, previous_analysis_script=prior_script,
                viewed_by_host_review_frames_0based=shown,script_sha256=code_hash,protocol_events=list(client.events)))
        retained_result=last_result
        if review.get('review_incomplete'):
            last_result['agent_record']['ai_review_status']='incomplete'
            if old is None: flag_all_for_review(last_result)
            break
        if review['decision']=='accept':
            last_result['agent_record']['ai_review_status']='accepted'
            break
        payload={**base_payload,'task':'Repair only the concrete problems in this review, starting from '
            f'checkpoint_{attempt+1}.json. Use load_checkpoint({attempt+1}) / save_checkpoint({attempt+1}, replacements=...) '
            'to retain correct observations, earlier repairs and frame assessments. Change IDs globally only where '
            'needed for temporal consistency; do not redetect unchanged frames. Execute the revised script and '
            'inspect changed overlays once. Keep uncertain cases flagged for human review.',
            'review':review,'latest_checkpoint':attempt+1}
    else:
        if last_result is None: raise ValueError('The AI did not produce a valid complete result: '+last_error)
        # Return the last measured result for review without claiming AI acceptance.
        last_result['warnings'].append('The AI requested further correction. Review this result or run Refine before Apply.')
        last_result['agent_record']['ai_review_status']='revision_requested'
        if old is None: flag_all_for_review(last_result)
    check();verify_inputs()
    report(100,'Agent analysis complete. Review the measured particle ROIs and IDs, then Apply.')
    return last_result
