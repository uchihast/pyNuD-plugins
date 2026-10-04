"""Export the current Particle Tracking observations on their captured images."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtWidgets

from background_auto import ProcessingCancelled, check_cancel


def frame_seconds(owner):
    """Use acquisition milliseconds, including restored AI input, never playback FPS."""
    import globalvals as gv
    if getattr(owner, '_ai_input_source', None):
        milliseconds = owner.ai_tracking_result['source'].header.frame_time
    else:
        milliseconds = getattr(owner, '_movie_acquisition', {}).get('frame_time_ms', gv.FrameTime)
    if not isinstance(milliseconds, (int, float)) or not np.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError('The acquisition frame time is missing. Reload the ASD and track again.')
    return float(milliseconds) / 1000.


def capture_acquisition(owner):
    """Called when manual Detect All Frames starts, before the main view can change."""
    import globalvals as gv
    from reference_profile import reference_file_identity
    owner._movie_acquisition = dict(source=reference_file_identity(gv), frame_time_ms=gv.FrameTime,
                                   scan_size=(gv.XScanSize, gv.YScanSize))


@dataclass
class MovieInput:
    frames: tuple
    detections: dict
    tracks: object
    scan_size: tuple
    seconds: float
    source: str
    palette: object = None
    display: object = None
    ai: bool = False


def snapshot(owner):
    """Detach edited tracks and retained detections from both main and AI review state."""
    import globalvals as gv
    from reference_profile import reference_file_identity
    current = reference_file_identity(gv)
    ai = bool(getattr(owner, '_ai_input_source', None))
    if ai:
        source = owner.ai_tracking_result['source']
        source.assert_unchanged()
        if Path(current[0]).resolve() != source.path:
            raise ValueError('Open the source ASD of this tracking result before exporting.')
        scan = owner._ai_scan_size
    else:
        acquisition = getattr(owner, '_movie_acquisition', {})
        if acquisition and acquisition['source'] != current:
            raise ValueError('The source ASD has changed. Detect and track its frames again.')
        scan = acquisition.get('scan_size', (owner.scan_size_x, owner.scan_size_y))
    stored = getattr(owner, 'frame_data_by_frame', {})
    if not stored or sorted(stored) != list(range(len(stored))):
        raise ValueError('Detect All Frames and track particles before exporting a movie.')
    frames = []
    for i in range(len(stored)):
        frame = stored[i]
        if (frame is None or np.ndim(frame) != 2 or min(frame.shape) < 2
                or not np.isfinite(frame).all() or (frames and frame.shape != frames[0].shape)):
            raise ValueError(f'Frame {i+1} has no valid tracking image. Detect All Frames again.')
        frames.append(np.array(frame, copy=True))
        frames[-1].flags.writeable = False
    tracks = getattr(owner, 'tracks_df', None)
    if tracks is None or tracks.empty:
        raise ValueError('No tracked particles to export.')
    if len(scan) != 2 or any(not np.isfinite(v) or v <= 0 for v in scan):
        raise ValueError('Invalid physical scan size.')
    tracks = tracks.copy(deep=True)
    if 'review_required' not in tracks:
        tracks['review_required'] = False
    detections = deepcopy(owner.particles_by_frame)
    # Manual linkers need not write IDs back to Particle objects. The edited
    # table is authoritative for both manual and AI trajectories.
    for index, particles in detections.items():
        rows = tracks[tracks.frame == index]
        lookup = {(float(r.x), float(r.y)): int(r.particle) for r in rows.itertuples()}
        for particle in particles:
            particle.track_id = lookup.get((particle.x, particle.y))
    return MovieInput(tuple(frames), detections, tracks, tuple(scan), frame_seconds(owner), str(current[0]),
                      deepcopy(getattr(owner, '_ai_palette', None)) if ai else None,
                      deepcopy(getattr(owner, '_ai_display_settings', None)) if ai else None, ai)


@dataclass(frozen=True)
class MovieOptions:
    first: int = 0
    last: int = 0
    fps: float = 10.
    edge: int = 720
    rois: bool = True
    centers: bool = True
    ids: bool = True
    trails: bool = True
    trail_length: int = 30
    time_label: bool = True


def render_frame(movie, index, options):
    """RGB movie pixels; input y increases upwards, pixel-edge ROIs stay intact."""
    frame = movie.frames[index]
    width_nm, height_nm = movie.scan_size
    scale = options.edge / max(width_nm, height_nm)
    width, height = max(2, round(width_nm*scale)), max(2, round(height_nm*scale))
    if movie.ai:
        from background_auto import preview_pixels
        if movie.display is not None:
            from main_view_data import display_tone_pixels
            gray = display_tone_pixels(frame, movie.display)
        else:
            low, high = np.percentile(frame, (1, 99))
            gray = np.uint8(np.clip((frame-low)/max(high-low, 1e-12), 0, 1)*255)
        rgb = preview_pixels(np.flipud(cv2.resize(gray, (width, height))), movie.palette)
    else:
        # Matches the manual panel's imshow(normalize per frame, viridis).
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        rgb = colormaps['viridis'](Normalize()(frame), bytes=True)[..., :3]
        rgb = cv2.resize(np.flipud(rgb), (width, height), interpolation=cv2.INTER_NEAREST)
    image = Image.new('RGB', (width + width % 2, height + height % 2 + (28 if options.time_label else 0)))
    image.paste(Image.fromarray(np.asarray(rgb, np.uint8)), (0, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    sx, sy = width/frame.shape[1], height/frame.shape[0]
    def center(x, y): return ((float(x)+.5)*sx, (frame.shape[0]-.5-float(y))*sy)
    if options.trails:
        lower = max(options.first, index-options.trail_length+1) if options.trail_length else options.first
        rows = movie.tracks[(movie.tracks.frame >= lower) & (movie.tracks.frame <= index)]
        colors = ('#37ff9c', '#61aaff', '#e4a0ff', '#ffb566', '#75e8ed')
        for ident, track in rows.groupby('particle'):
            previous = None
            for row in track.sort_values('frame').itertuples():
                point = center(row.x, row.y)
                # Gaps have no measured path: do not bridge them with a line.
                if previous is not None and row.frame == previous[0]+1:
                    draw.line((previous[1], point), fill=colors[int(ident) % len(colors)], width=2)
                previous = (row.frame, point)
    selected = movie.tracks[movie.tracks.frame == index]
    flagged = set(selected.loc[selected.review_required, 'particle'])
    for number, p in enumerate(movie.detections.get(index, []), 1):
        color = '#ff5858' if p.size_unresolved or p.track_id in flagged else '#ffd447' if p.track_id is None else '#37ff9c'
        x, y = center(p.x, p.y)
        bounds = p.roi or (max(0, p.x+.5-p.radius), max(0, p.y+.5-p.radius),
                          min(frame.shape[1], p.x+.5+p.radius), min(frame.shape[0], p.y+.5+p.radius))
        left, bottom, right, top = bounds
        box = (left*sx, (frame.shape[0]-top)*sy, min(width-1, right*sx), min(height-1, (frame.shape[0]-bottom)*sy))
        if options.rois:
            draw.rectangle(box, outline='black', width=4)
            draw.rectangle(box, outline=color, width=2)
        if options.centers:
            for thickness, ink in ((4, 'black'), (2, color)):
                draw.line((x-4, y, x+4, y), fill=ink, width=thickness)
                draw.line((x, y-4, x, y+4), fill=ink, width=thickness)
        if options.ids:
            label = f'ID {p.track_id}' if p.track_id is not None else f'D{number}'
            text_box = draw.textbbox((0, 0), label, font=font)
            tw, th = text_box[2]-text_box[0], text_box[3]-text_box[1]
            tx, ty = max(0, min(box[0], width-tw-6)), max(0, min(box[1]-th-6, height-th-6))
            draw.rectangle((tx, ty, tx+tw+5, ty+th+5), fill='black')
            draw.text((tx+2, ty+2-text_box[1]), label, fill=color, font=font)
    if options.time_label:
        draw.text((6, height+5), f'Frame {index+1}/{len(movie.frames)}   t = {index*movie.seconds:.6g} s', fill='white', font=font)
    return np.asarray(image)


def export_movie(movie, path, options, cancelled=lambda: False, progress=lambda n, text: None):
    """Stage, verify, and publish one movie; cancellation preserves existing output."""
    path = Path(path)
    if path.suffix.lower() not in ('.mp4', '.avi'):
        raise ValueError('Choose an .mp4 or .avi movie file.')
    if (not 0 <= options.first <= options.last < len(movie.frames)
            or not np.isfinite(options.fps) or not .01 <= options.fps <= 1000
            or not 160 <= options.edge <= 4096 or options.trail_length < 0):
        raise ValueError('Invalid movie frame range, FPS or image size.')
    check_cancel(cancelled)
    fd, name = tempfile.mkstemp(prefix='.pynud-particles-', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    writer = None
    try:
        first = render_frame(movie, options.first, options)
        size = (first.shape[1], first.shape[0])
        codecs = ('avc1', 'mp4v') if path.suffix.lower() == '.mp4' else ('MJPG',)
        for codec in codecs:
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), options.fps, size)
            if writer.isOpened(): break
            writer.release(); writer = None
        if writer is None:
            raise ValueError('No encoder is available for this format. Try the other movie format.')
        count = options.last-options.first+1
        for offset, index in enumerate(range(options.first, options.last+1)):
            check_cancel(cancelled)
            rgb = first if offset == 0 else render_frame(movie, index, options)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            progress(round(90*(offset+1)/count), f'Writing frame {index+1}/{len(movie.frames)}…')
        writer.release(); writer = None
        capture = cv2.VideoCapture(str(temporary))
        try:
            for offset in range(count):
                check_cancel(cancelled)
                ok, frame = capture.read()
                if not ok or (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError('The encoded movie failed frame validation. No movie was saved.')
                progress(90+round(9*(offset+1)/count), f'Checking saved movie {offset+1}/{count}…')
        finally:
            capture.release()
        check_cancel(cancelled)
        os.replace(temporary, path)
        progress(100, 'Saved: ' + str(path))
        return str(path)
    finally:
        if writer is not None: writer.release()
        temporary.unlink(missing_ok=True)


class MovieWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    result = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, movie, path, options, parent):
        super().__init__(parent)
        self.movie, self.path, self.options = movie, path, options

    def run(self):
        try:
            self.result.emit(export_movie(self.movie, self.path, self.options, self.isInterruptionRequested, self.progress.emit))
        except ProcessingCancelled:
            self.failed.emit('Export cancelled. No movie was saved.')
        except Exception as exc:
            self.failed.emit('Export failed: ' + str(exc))


class ParticleMovieDialog(QtWidgets.QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.movie = snapshot(owner)
        self.worker = None
        self.setWindowTitle('Export Particle Tracking Movie')
        self.resize(820, 720)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Current tracking results, including manual edits. No tracking is recalculated.\n'
            'Red: review required. Yellow: unlinked. ROI uses saved bounds, or the detection radius when bounds are unavailable.')
        note.setWordWrap(True); layout.addWidget(note)
        form = QtWidgets.QGridLayout(); layout.addLayout(form)
        self.first = QtWidgets.QSpinBox(); self.last = QtWidgets.QSpinBox()
        for widget in (self.first, self.last): widget.setRange(1, len(self.movie.frames))
        self.last.setValue(len(self.movie.frames))
        self.fps = QtWidgets.QDoubleSpinBox(); self.fps.setRange(.01, 1000); self.fps.setDecimals(4)
        self.fps.setValue(1/self.movie.seconds)
        self.fps.setToolTip('Playback FPS only; acquisition times in the movie remain unchanged.')
        self.edge = QtWidgets.QSpinBox(); self.edge.setRange(160, 4096); self.edge.setSingleStep(80); self.edge.setValue(720)
        for row, (text, widget) in enumerate((('From frame', self.first), ('Through frame', self.last),
                                             ('Playback FPS', self.fps), ('Image long edge (px)', self.edge))):
            form.addWidget(QtWidgets.QLabel(text), row//2, (row%2)*2); form.addWidget(widget, row//2, (row%2)*2+1)
        self.checks = {}
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        for key, label in (('rois','ROI'), ('centers','Centers'), ('ids','IDs'), ('trails','Trails'), ('time_label','Frame / Time')):
            box = QtWidgets.QCheckBox(label); box.setChecked(True); row.addWidget(box); self.checks[key] = box
            box.toggled.connect(self.preview)
        self.checks['trails'].setChecked(owner.show_tracks_check.isChecked())
        self.checks['ids'].setChecked(owner.show_track_ids_check.isChecked())
        self.length = QtWidgets.QSpinBox(); self.length.setRange(0, len(self.movie.frames)); self.length.setValue(min(30, len(self.movie.frames)))
        self.length.setSpecialValueText('All previous'); row.addWidget(QtWidgets.QLabel('Trail frames')); row.addWidget(self.length)
        self.image = QtWidgets.QLabel(); self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setMinimumSize(200, 180); self.image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image.setStyleSheet('background: black'); layout.addWidget(self.image, 1)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, len(self.movie.frames)-1); layout.addWidget(self.slider)
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0); layout.addWidget(self.bar)
        self.status = QtWidgets.QLabel(f'Acquisition interval: {self.movie.seconds:g} s/frame. Preview uses the selected overlays.')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout(); layout.addLayout(buttons)
        self.save = QtWidgets.QPushButton('Save Movie…'); self.close_button = QtWidgets.QPushButton('Close')
        buttons.addWidget(self.save); buttons.addWidget(self.close_button)
        self.save.clicked.connect(self.start); self.close_button.clicked.connect(self.close)
        for widget in (self.first, self.last, self.edge, self.length, self.slider): widget.valueChanged.connect(self.preview)
        self.preview()

    def options(self):
        return MovieOptions(self.first.value()-1, self.last.value()-1, self.fps.value(), self.edge.value(),
                            **{key: box.isChecked() for key, box in self.checks.items()}, trail_length=self.length.value())

    def preview(self, *_):
        if not hasattr(self, 'slider'): return
        try:
            rgb = np.ascontiguousarray(render_frame(self.movie, self.slider.value(), self.options()))
            image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        except Exception as exc:
            self.status.setText(str(exc))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.preview()

    def start(self):
        if self.worker: return
        options = self.options()
        if options.first > options.last:
            self.status.setText('From frame must not exceed Through frame.'); return
        default = Path(self.movie.source).with_name(Path(self.movie.source).stem + datetime.now().strftime('_%Y%m%d_tracking.mp4'))
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, 'Save tracking movie', str(default),
                                                              'MP4 movie (*.mp4);;AVI movie (*.avi)')
        if not path: return
        if not Path(path).suffix: path += '.avi' if selected.startswith('AVI') else '.mp4'
        self.worker = MovieWorker(self.movie, path, options, self)
        self.save.setEnabled(False); self.close_button.setText('Cancel export')
        for widget in (self.first,self.last,self.fps,self.edge,self.length,*self.checks.values()): widget.setEnabled(False)
        self.worker.progress.connect(self.progress)
        self.worker.result.connect(lambda name: self.status.setText('Saved: ' + name))
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def progress(self, value, text):
        self.bar.setValue(value); self.status.setText(text)

    def finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.save.setEnabled(True); self.close_button.setText('Close')
        for widget in (self.first,self.last,self.fps,self.edge,self.length,*self.checks.values()): widget.setEnabled(True)

    def closeEvent(self, event):
        if self.worker:
            self.worker.requestInterruption(); self.status.setText('Cancelling export…'); event.ignore(); return
        event.accept()

    def reject(self): self.close()
