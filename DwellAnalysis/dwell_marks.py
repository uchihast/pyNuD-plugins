"""Shared, transactional frame marks for manual and AI dwell reviews."""
from copy import deepcopy
import math

import numpy as np
from PIL import ImageDraw
from analysis_input import source_unchanged
from background_auto import check_cancel
from . import dwell_ai as ai

STATES = ('present', 'absent', 'unknown')


class MarkDraft:
    """Stable ROI IDs with explicit unknowns; editing never changes the source review."""
    def __init__(self, session):
        self.session = session
        self.count = len(session['snapshots'])
        self.shape = session['snapshots'][0].image.shape
        self.records = {}; self.changed = set(); self.history = []
        result = session['result']; assessed = set(result['assessed_frames'])
        self.replaced = {s['id']: s.get('manual_replaced_ranges', []) for s in result['sites']}
        for site in result['sites']:
            first, last = site['first_frame'], site['last_frame']
            indices = np.clip(np.arange(self.count)+1, first, last)-first
            states = ['unknown']*self.count; reviewed = [False]*self.count; origins = ['unreviewed']*self.count
            events = [e for e in result['events'] if e['site'] == site['id']]
            for frame in range(first, last+1):
                k, i = frame-1, frame-first
                if self.is_replaced(site['id'], frame): continue
                if 'mark_states' in site:
                    states[k] = site['mark_states'][i]; reviewed[k] = site['mark_reviewed'][i]
                    origins[k] = site['mark_sources'][i]
                    continue
                episode = any(e['first_frame'] <= frame <= e['last_frame'] for e in events)
                if session['origin'] == 'manual':
                    if frame in assessed:
                        states[k] = 'present' if episode else 'absent'
                        reviewed[k] = True; origins[k] = 'manual'
                else:
                    present = site.get('particle_present', [False]*len(site['valid']))[i]
                    if present or episode:
                        states[k] = 'present'; origins[k] = 'ai'
                    elif site['valid'][i] and site.get('low_nm') is not None and site['response_nm'][i] < site['low_nm']:
                        states[k] = 'absent'; origins[k] = 'ai'
            self.records[site['id']] = dict(states=states, reviewed=reviewed, origins=origins,
                positions=np.asarray(site['positions_px'])[indices].tolist(), radius=float(site['radius_px']))

    def _range(self, first, last):
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= self.count:
            raise ValueError('Choose frames inside the captured movie.')

    def is_replaced(self, ident, frame):
        return any(a <= frame <= b for a, b in self.replaced.get(ident, []))

    def _check_site(self, ident, first, last):
        self._range(first, last)
        if ident not in self.records: raise ValueError('Select a particle ID.')
        site = next((s for s in self.session['result']['sites'] if s['id'] == ident), {})
        if any(a <= last and b >= first for a, b in site.get('manual_replaced_ranges', [])):
            raise ValueError('These frames were replaced by split/moved ROIs. Edit the replacement ID instead.')

    def _checkpoint(self, ids):
        self.history.append((self.changed.copy(), {i: deepcopy(self.records.get(i)) for i in ids}))

    def undo(self):
        if not self.history: return
        self.changed, records = self.history.pop()
        for ident, record in records.items():
            if record is None: self.records.pop(ident, None)
            else: self.records[ident] = record

    def set_state(self, ident, first, last, state):
        self._check_site(ident, first, last)
        if state not in STATES: raise ValueError('Invalid mark state.')
        self._checkpoint([ident]); self.changed.add(ident); record = self.records[ident]
        for i in range(first-1, last):
            record['states'][i] = state; record['reviewed'][i] = state != 'unknown'; record['origins'][i] = 'manual'

    def move(self, ident, frame, x, y):
        self._check_site(ident, frame, frame); self._point(x, y)
        self._checkpoint([ident]); self.changed.add(ident)
        record = self.records[ident]; i = frame-1
        record['positions'][i] = [float(x), float(y)]
        record['states'][i] = 'present'; record['reviewed'][i] = True; record['origins'][i] = 'manual'

    def _point(self, x, y):
        h, w = self.shape
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h):
            raise ValueError('Click a center inside the image.')

    def add(self, frame, x, y, radius):
        self._range(frame, frame); self._point(x, y)
        radius = float(radius)
        if len(self.records) >= 2000: raise ValueError('This review already contains 2000 particle IDs.')
        if not math.isfinite(radius) or not .5 <= radius <= min(self.shape)/4:
            raise ValueError('Choose a valid mark radius.')
        serial = 1
        while f'Mark{serial:03d}' in self.records: serial += 1
        ident = f'Mark{serial:03d}'; self._checkpoint([ident]); self.changed.add(ident)
        record = dict(states=['unknown']*self.count, reviewed=[False]*self.count,
            origins=['unreviewed']*self.count, positions=[[float(x), float(y)] for _ in range(self.count)], radius=radius)
        record['states'][frame-1] = 'present'; record['reviewed'][frame-1] = True; record['origins'][frame-1] = 'manual'
        self.records[ident] = record
        return ident

    def memorize(self, frame):
        self._range(frame, frame)
        ids = [i for i, r in self.records.items() if r['states'][frame-1] != 'unknown' and not self.is_replaced(i, frame)]
        for ident in ids: self._check_site(ident, frame, frame)
        if not ids: return
        self._checkpoint(ids)
        for ident in ids:
            self.changed.add(ident); r = self.records[ident]
            r['reviewed'][frame-1] = True; r['origins'][frame-1] = 'manual'

    def restore_previous(self, frame):
        self._range(frame, frame)
        if frame <= 1: return
        ids = [i for i, r in self.records.items() if r['states'][frame-2] != 'unknown'
               and not self.is_replaced(i, frame) and not self.is_replaced(i, frame-1)]
        for ident in ids: self._check_site(ident, frame, frame)
        if not ids: return
        self._checkpoint(ids)
        for ident in ids:
            self.changed.add(ident); r = self.records[ident]
            r['states'][frame-1] = r['states'][frame-2]
            r['positions'][frame-1] = list(r['positions'][frame-2])
            # Copying a mark is not an observation of the next frame.
            r['reviewed'][frame-1] = False; r['origins'][frame-1] = 'restored'

    def render(self, frame, selected=None):
        image = ai.render_frame(self.session['snapshots'][frame-1]); draw = ImageDraw.Draw(image)
        h, w = self.shape; sx, sy = image.width/w, image.height/h
        for ident, r in self.records.items():
            if self.is_replaced(ident, frame): continue
            i = frame-1; state = r['states'][i]
            if state != 'present' and ident != selected: continue
            x, y = r['positions'][i]; cx, cy = (x+.5)*sx, (h-.5-y)*sy; radius = r['radius']
            color = '#999999' if state == 'absent' else '#43a5ff' if state == 'unknown' else '#39eead' if r['reviewed'][i] else '#ffbb55'
            draw.ellipse((cx-radius*sx, cy-radius*sy, cx+radius*sx, cy+radius*sy), outline=color, width=3 if ident == selected else 2)
            draw.line((cx-4, cy, cx+4, cy), fill=color); draw.line((cx, cy-4, cx, cy+4), fill=color)
            draw.text((cx+radius*sx+2, cy-radius*sy), ident, fill=color)
        return image


def empty_session(snapshots, dt):
    return ai.make_session(snapshots, dt, dict(algorithm='Manual frame marks',
        signal_definition='Local ROI height minus surrounding background (nm).', warnings=[],
        assessed_frames=[], sites=[], events=[], parameters={}), 'manual')


def commit_marks(draft, progress=lambda p, m: None, cancelled=lambda: False):
    """Remeasure edited IDs only; convert marked runs to pending, censored candidates."""
    if not draft.changed: return draft.session
    session = draft.session; snapshots = session['snapshots']; n = len(snapshots); dt = session['frame_time_s']
    result = ai.validate_result(session['result'], snapshots, dt, decisions=True)
    original_sites = {s['id']: s for s in result['sites']}; h, w = draft.shape; yy, xx = np.indices((h, w))
    # Retain unrelated IDs, event choices and their exact numerical method.
    retained_events = [e for e in result['events'] if e['site'] not in draft.changed]
    used_ids = {e['id'] for e in result['events']}; assessed = set(result['assessed_frames'])
    exclusions = result.get('event_exclusion_ranges', [])
    ids = list(draft.records)
    all_positions = np.asarray([draft.records[i]['positions'] for i in ids])
    all_present = np.asarray([[s == 'present' for s in draft.records[i]['states']] for i in ids])
    all_radii = np.asarray([draft.records[i]['radius'] for i in ids])
    for ident in sorted(draft.changed):
        for k, observed in enumerate(draft.records[ident]['reviewed']):
            if observed: assessed.add(k+1)
    updates = {}
    for index, ident in enumerate(sorted(draft.changed)):
        check_cancel(cancelled); record = draft.records[ident]; old = original_sites.get(ident)
        states, reviewed = record['states'], record['reviewed']; radius = record['radius']; self_index = ids.index(ident)
        signal = []; valid = []; flags = []
        for k, (snap, (x, y)) in enumerate(zip(snapshots, record['positions'])):
            check_cancel(cancelled); source_unchanged(snap)
            d2 = (xx-x)**2+(yy-y)**2; disk = d2 <= radius**2; ring = (d2 >= (radius*1.5)**2) & (d2 <= (radius*2.5)**2)
            distances = np.sum((all_positions[:, k]-[x, y])**2, axis=1)
            neighbours = all_present[:, k] & (distances <= (2.5*radius+1.5*all_radii)**2)
            neighbours[self_index] = False
            for other in np.flatnonzero(neighbours):
                ox, oy = all_positions[other, k]
                ring &= (xx-ox)**2+(yy-oy)**2 > (all_radii[other]*1.5)**2
            baseline = float(np.median(snap.image[ring])) if ring.any() else 0.
            signal.append(float(np.percentile(snap.image[disk], 80)-baseline) if disk.any() else 0.)
            oi = k+1-old['first_frame'] if old else -1
            supported_before = old is not None and 0 <= oi < len(old['valid']) and old['valid'][oi]
            measured = (reviewed[k] or supported_before) and radius <= x < w-radius and radius <= y < h-radius and ring.sum() >= 8 and disk.any()
            measured &= k+1 in assessed and not any(a <= k+1 <= b for a, b in exclusions)
            measured &= not any(a <= k+1 <= b for a, b in (old or {}).get('manual_replaced_ranges', []))
            valid.append(bool(measured))
            flags.append(['frame_mark']+([] if measured else ['unsupported_mark_measurement']))
        site = dict(id=ident, first_frame=1, last_frame=n, radius_px=radius, display_radius_px=radius*1.5,
            positions_px=deepcopy(record['positions']), measurement_positions_px=deepcopy(record['positions']),
            response_nm=signal, valid=valid, particle_present=[s == 'present' for s in states], particle_flags=flags,
            low_nm=None, high_nm=None, review_decision='pending', mark_states=list(states),
            mark_reviewed=list(reviewed), mark_sources=list(record['origins']),
            signal_definition='Frame marks: 80th percentile within the marked circle minus the median in its '
                '1.5–2.5 radius annulus, excluding other marked particles (nm). Events follow presence marks, '
                'not height thresholds; presence does not by itself establish binding.',
            manual_parameters=dict(method='frame_marks', measurement='local_height_annulus', version=1))
        if old:
            for key in ('source_site', 'manual_replaced_ranges'):
                if key in old: site[key] = deepcopy(old[key])
        updates[ident] = site
        # Only measured present marks form runs. Unknowns and unsupported frames
        # split runs but never assert a disappearance/reappearance.
        runs = []
        for k, state in enumerate(states):
            if state == 'present' and valid[k]:
                if runs and runs[-1][1] == k-1: runs[-1][1] = k
                else: runs.append([k, k])
        for a, b in runs:
            serial = 1
            while f'MarkE{serial:04d}' in used_ids: serial += 1
            eid = f'MarkE{serial:04d}'; used_ids.add(eid)
            left = a == 0 or states[a-1] != 'absent' or not valid[a-1]
            right = b == n-1 or states[b+1] != 'absent' or not valid[b+1]
            unreviewed = not all(reviewed[max(0,a-1):min(n,b+2)])
            retained_events.append(dict(id=eid, site=ident, first_frame=a+1, last_frame=b+1,
                status='review_required' if unreviewed else 'manual', decision='pending',
                left_censored=left, right_censored=right,
                flags=['frame_marks', 'presence_requires_event_review']+(['unreviewed_marks'] if unreviewed else [])))
        progress(round(95*(index+1)/len(draft.changed)), f'Remeasuring frame marks: {ident}…')
    check_cancel(cancelled)
    result['sites'] = [updates.pop(s['id'], s) for s in result['sites']]+list(updates.values())
    result['events'] = retained_events; result['assessed_frames'] = sorted(assessed)
    result['warnings'] = list(dict.fromkeys(result['warnings']+[
        'Edited frame marks form candidate residence intervals. Unknown/unsupported gaps are censored, not observed unbinding. '
        'Review event boundaries and accept explicitly before Apply.']))
    checked = ai.validate_result(result, snapshots, dt, decisions=True)
    audit = deepcopy(session.get('audit', {})); history = audit.pop('refinement_history', [])
    history.append(dict(result=deepcopy(session['result']), audit=deepcopy(audit), origin=session['origin']))
    audit['refinement_history'] = history
    audit.setdefault('mark_edits', []).append(dict(site_ids=sorted(draft.changed), method='frame_marks',
        previous_events=[deepcopy(e) for e in session['result']['events'] if e['site'] in draft.changed]))
    audit['ai_review_status'] = 'manual_marks_edited'
    progress(100, 'Frame marks measured. Review the revised events before Apply.')
    return dict(session, result=checked, audit=audit)
