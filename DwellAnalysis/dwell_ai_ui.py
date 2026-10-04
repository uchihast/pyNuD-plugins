"""Common movie / event table / height trace review for Dwell Analysis."""
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from ai_data_sharing import AISharingConsent, current_background_source
from ai_providers import load_provider_settings
from ai_progress import AIStopNotice, AIProgress
from analysis_runtime import agent_availability, create_agent_client
from analysis_input import source_unchanged
from . import dwell_ai as backend
from . import dwell_marks
class DwellWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, job, parent):
        super().__init__(parent); self.job = job
        from ai_recovery import attach_recovery
        recovery = attach_recovery(self, parent)
        recovery.paused.connect(parent.pause_for_retry); recovery.resumed.connect(parent.resumed)
        recovery.activity.connect(parent.observed_activity)

    def run(self):
        try: self.completed.emit(self.job(self.progress.emit))
        except Exception as exc: self.failed.emit(str(exc) or 'Cancelled. Existing results retained.')


class DwellImage(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)
    def __init__(self):
        super().__init__(); self.pixmap = QtGui.QPixmap(); self.shape = None
        self.setMinimumSize(180, 140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_image(self, image, shape=None):
        self.shape = shape
        data = image.convert('RGB'); array = np.ascontiguousarray(data)
        qimage = QtGui.QImage(array.data, data.width, data.height, array.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.pixmap = QtGui.QPixmap.fromImage(qimage); self.update()

    def image_rect(self):
        size = self.pixmap.size(); size.scale(self.size(), QtCore.Qt.KeepAspectRatio)
        return QtCore.QRect((self.width()-size.width())//2, (self.height()-size.height())//2, size.width(), size.height())

    def mousePressEvent(self, event):
        if self.shape is None or self.pixmap.isNull() or event.button() != QtCore.Qt.LeftButton: return
        rect = self.image_rect()
        if not rect.contains(event.pos()): return
        h, w = self.shape
        x = (event.x()-rect.x())*w/rect.width()-.5
        y = h-.5-(event.y()-rect.y())*h/rect.height()
        self.clicked.emit(float(np.clip(x, 0, w-1)), float(np.clip(y, 0, h-1)))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self); painter.fillRect(self.rect(), QtGui.QColor('#111722'))
        if not self.pixmap.isNull():
            image = self.pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            painter.drawPixmap((self.width()-image.width())//2, (self.height()-image.height())//2, image)
        else:
            painter.setPen(QtGui.QColor('white'))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, 'Run AI Analysis or finalize manual marks to review.')


class ROICorrectionDialog(QtWidgets.QDialog):
    def __init__(self, session, site, frame, parent):
        super().__init__(parent); self.session=session; self.site=site; self.frame=frame; self.centers=[]
        adding = site is None
        self.setWindowTitle(('Add ROIs' if adding else 'Correct ROI')+' — click particle centers'); self.resize(850,740)
        self.setMinimumSize(520,480); layout=QtWidgets.QVBoxLayout(self)
        instructions = (f'Add missed particles · Frame {frame}\n'
            'Click each missed particle center. Each point creates a new ROI; existing ROIs stay unchanged. '
            'Check the existing boxes to avoid adding the same particle twice. ' if adding else
            f'ROI {site["id"]} · Frame {frame}\n'
            'Click one center to move the ROI, or multiple centers to split it. '
            'Clicks replace this ROI only, from this frame through the chosen last frame. ')
        text=QtWidgets.QLabel(instructions+'Local tracking and height measurement run from this frame through the chosen last frame '
            'without AI; new events require review.')
        text.setWordWrap(True); layout.addWidget(text)
        self.preview=DwellImage(); self.preview.clicked.connect(self.add_center); layout.addWidget(self.preview,1)
        row=QtWidgets.QHBoxLayout(); self.count=QtWidgets.QLabel(); row.addWidget(self.count,1)
        undo=QtWidgets.QPushButton('Undo last point'); undo.clicked.connect(self.undo); row.addWidget(undo)
        clear=QtWidgets.QPushButton('Clear points'); clear.clicked.connect(self.clear); row.addWidget(clear); layout.addLayout(row)
        last_frame = len(session['snapshots']) if adding else site['last_frame']
        form=QtWidgets.QFormLayout(); self.last=QtWidgets.QSpinBox(); self.last.setRange(frame,last_frame); self.last.setValue(last_frame)
        form.addRow('Last frame to remeasure',self.last)
        limit=min(session['snapshots'][frame-1].image.shape)/4
        self.radius=QtWidgets.QDoubleSpinBox(); self.radius.setRange(.5,limit); self.radius.setDecimals(1)
        default_radius = float(np.median([s['radius_px'] for s in session['result']['sites']])) if adding and session['result']['sites'] else 3.
        self.radius.setValue(min(limit,default_radius if adding else site['radius_px'])); self.radius.setSuffix(' px')
        form.addRow('Height sampling radius',self.radius)
        self.step=QtWidgets.QDoubleSpinBox(); self.step.setRange(0,limit); self.step.setValue(min(3.,limit)); self.step.setSuffix(' px/frame')
        self.step.setToolTip('Maximum local motion. Use 0 for fixed centers. Weak matches are excluded from event timing.')
        form.addRow('Tracking search distance',self.step); layout.addLayout(form)
        note=QtWidgets.QLabel('Sampling circles must not overlap. Heights are remeasured from the captured main image; '
            'edits are recorded in JSON history. No original ASD is changed.')
        note.setWordWrap(True); layout.addWidget(note)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        self.ok=buttons.button(QtWidgets.QDialogButtonBox.Ok); self.ok.setText('Add and measure ROIs' if adding else 'Remeasure ROI')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.radius.valueChanged.connect(self.redraw); self.redraw()

    def add_center(self,x,y):
        if len(self.centers)<20: self.centers.append([x,y]); self.redraw()
    def undo(self):
        if self.centers: self.centers.pop(); self.redraw()
    def clear(self):
        self.centers=[]; self.redraw()
    def redraw(self):
        from PIL import ImageDraw
        snap=self.session['snapshots'][self.frame-1]; h,w=snap.image.shape
        image=backend.render_frame(snap,self.session['result'],self.frame,self.site['id'] if self.site else None)
        draw=ImageDraw.Draw(image); sx=image.width/w; sy=image.height/h
        for i,(x,y) in enumerate(self.centers,1):
            cx,cy=(x+.5)*sx,(h-.5-y)*sy; radius=self.radius.value()
            draw.ellipse((cx-radius*sx,cy-radius*sy,cx+radius*sx,cy+radius*sy),outline='#ff48f0',width=2)
            draw.line((cx-5,cy,cx+5,cy),fill='white',width=2); draw.line((cx,cy-5,cx,cy+5),fill='white',width=2)
            draw.text((cx+5,cy+5),str(i),fill='white')
        self.preview.set_image(image,snap.image.shape)
        overlap=any(np.linalg.norm(np.array(a)-b)<=2*self.radius.value() for i,a in enumerate(self.centers) for b in self.centers[i+1:])
        self.count.setText(f'{len(self.centers)} center(s)'+(' — circles overlap; reduce radius' if overlap else ''))
        self.ok.setEnabled(bool(self.centers) and not overlap)


class DwellRefineDialog(QtWidgets.QDialog):
    def __init__(self, frame, site, parent):
        super().__init__(parent); self.setWindowTitle('Refine Dwell Analysis with AI'); self.resize(650, 330)
        layout = QtWidgets.QVBoxLayout(self)
        context = QtWidgets.QLabel(f'Context: Frame {frame}'+(f' · ROI {site}' if site else '')+
            '\nAI receives the current movie, detections, traces and review choices. Describe false detections, '
            'missed particles, wrong ROI positions or event boundaries. It can revise detections and measurements '
            'throughout this movie; the displayed frame is the starting point for inspection.')
        context.setWordWrap(True); layout.addWidget(context)
        self.instructions = QtWidgets.QPlainTextEdit(); self.instructions.setObjectName('dwell_refine_instructions')
        self.instructions.setPlaceholderText('Example: In this frame, background texture and horizontal streaks are '
            'being counted as particles. Recheck the visible particle footprints and neighbouring frames, '
            'then remove false detections with evidence.')
        layout.addWidget(self.instructions, 1)
        note = QtWidgets.QLabel('The current result remains available until a validated revision is ready. '
            'Revised ROIs and events require review again. Undo Refine restores the previous result.')
        note.setWordWrap(True); layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        self.run_button = buttons.button(QtWidgets.QDialogButtonBox.Ok); self.run_button.setText('Refine with AI')
        self.run_button.setEnabled(False)
        self.instructions.textChanged.connect(lambda: self.run_button.setEnabled(bool(self.instructions.toPlainText().strip())))
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)


class FrameMarksDialog(QtWidgets.QDialog):
    """One mark editor for AI proposals, legacy marks and new manual reviews."""
    def __init__(self, session, frame, site, parent):
        super().__init__(parent)
        self.setWindowTitle('Dwell Analysis — Edit Frame Marks')
        self.resize(1120, 800); self.setMinimumSize(620, 480); self.setSizeGripEnabled(True)
        self.draft = dwell_marks.MarkDraft(session); self.selected = site
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Edit particle presence frame by frame. Green: manually confirmed present; orange: proposal; '
            'blue: unreviewed. Removing a mark means Absent; Unreviewed leaves a gap. '
            'Presence alone does not confirm binding. Revised events must be reviewed before Apply.')
        note.setWordWrap(True); layout.addWidget(note)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); layout.addWidget(scroll, 1)
        content = QtWidgets.QWidget(); content.setMinimumWidth(840); scroll.setWidget(content)
        body = QtWidgets.QVBoxLayout(content)
        modes = QtWidgets.QHBoxLayout(); body.addLayout(modes)
        modes.addWidget(QtWidgets.QLabel('Click action'))
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(['Select particle', 'Add particle', 'Move / mark selected', 'Mark clicked particle absent'])
        modes.addWidget(self.mode); modes.addWidget(QtWidgets.QLabel('New mark radius (px)'))
        self.radius = QtWidgets.QDoubleSpinBox(); self.radius.setRange(.5, min(self.draft.shape)/4)
        self.radius.setValue(min(3., min(self.draft.shape)/4)); modes.addWidget(self.radius); modes.addStretch()
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal); split.setChildrenCollapsible(False); split.setHandleWidth(8)
        body.addWidget(split, 1)
        self.preview = DwellImage(); self.preview.setMinimumSize(300, 260); split.addWidget(self.preview)
        self.preview.clicked.connect(self.image_clicked)
        panel = QtWidgets.QWidget(); right = QtWidgets.QVBoxLayout(panel)
        self.ids = QtWidgets.QComboBox(); self.ids.currentIndexChanged.connect(self.select_id); right.addWidget(self.ids)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Frame', 'Presence', 'Checked', 'Source'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.table_selected); right.addWidget(self.table)
        split.addWidget(panel); split.setSizes([650, 350])
        nav = QtWidgets.QHBoxLayout(); body.addLayout(nav)
        previous = QtWidgets.QPushButton('−1'); previous.clicked.connect(lambda: self.frame.setValue(self.frame.value()-1)); nav.addWidget(previous)
        following = QtWidgets.QPushButton('+1'); following.clicked.connect(lambda: self.frame.setValue(self.frame.value()+1)); nav.addWidget(following)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(1, self.draft.count); nav.addWidget(self.slider, 1)
        self.frame = QtWidgets.QSpinBox(); self.frame.setRange(1, self.draft.count); self.frame.setValue(frame); nav.addWidget(self.frame)
        self.slider.valueChanged.connect(self.frame.setValue); self.frame.valueChanged.connect(self.frame_changed)
        scope = QtWidgets.QHBoxLayout(); body.addLayout(scope)
        self.use_range = QtWidgets.QCheckBox('Use frame range'); scope.addWidget(self.use_range)
        self.first = QtWidgets.QSpinBox(); self.last = QtWidgets.QSpinBox()
        for spin in (self.first, self.last): spin.setRange(1, self.draft.count); spin.setValue(frame); spin.setEnabled(False); scope.addWidget(spin)
        self.use_range.toggled.connect(lambda checked: [spin.setEnabled(checked) for spin in (self.first, self.last)])
        for title, state in [('Present', 'present'), ('Absent', 'absent'), ('Unreviewed', 'unknown')]:
            button = QtWidgets.QPushButton(title); button.clicked.connect(lambda _, value=state: self.set_state(value)); scope.addWidget(button)
        scope.addStretch()
        commands = QtWidgets.QHBoxLayout(); body.addLayout(commands)
        self.memorize = QtWidgets.QPushButton('Memorize Frame'); self.memorize.clicked.connect(lambda: self.change(lambda: self.draft.memorize(self.frame.value())))
        commands.addWidget(self.memorize)
        self.restore = QtWidgets.QPushButton('Restore Previous'); self.restore.clicked.connect(lambda: self.change(lambda: self.draft.restore_previous(self.frame.value())))
        commands.addWidget(self.restore)
        self.undo = QtWidgets.QPushButton('Undo Mark Edit'); self.undo.clicked.connect(lambda: self.change(self.draft.undo)); commands.addWidget(self.undo)
        self.feedback = QtWidgets.QLabel(); self.feedback.setWordWrap(True); layout.addWidget(self.feedback)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.save = buttons.addButton('Review Revised Events', QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        for button in self.findChildren(QtWidgets.QPushButton): button.setAutoDefault(False); button.setDefault(False)
        self.refresh()

    def change(self, action):
        try:
            action(); self.refresh()
        except ValueError as exc:
            self.feedback.setText(str(exc))

    def refresh(self):
        if self.selected not in self.draft.records: self.selected = next(iter(self.draft.records), None)
        with QtCore.QSignalBlocker(self.ids):
            self.ids.clear()
            for ident in self.draft.records: self.ids.addItem(ident, ident)
            self.ids.setCurrentIndex(self.ids.findData(self.selected))
        record = self.draft.records.get(self.selected)
        with QtCore.QSignalBlocker(self.table):
            self.table.setRowCount(self.draft.count if record else 0)
            if record:
                for k, state in enumerate(record['states']):
                    values = [k+1, {'present': 'Present', 'absent': 'Absent', 'unknown': 'Unreviewed'}[state],
                              'Yes' if record['reviewed'][k] else 'No', record['origins'][k]]
                    for col, value in enumerate(values): self.table.setItem(k, col, QtWidgets.QTableWidgetItem(str(value)))
        self.frame_changed(self.frame.value())
        self.undo.setEnabled(bool(self.draft.history)); self.save.setEnabled(bool(self.draft.changed))
        self.feedback.setText(f'{len(self.draft.changed)} edited IDs. Only edited IDs will be remeasured; '
            'their events return to pending. Cancel discards these edits. Restore copies proposals; Memorize confirms the displayed states.')

    def select_id(self):
        self.selected = self.ids.currentData(); self.refresh()

    def table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows: self.frame.setValue(rows[0].row()+1)

    def frame_changed(self, value):
        with QtCore.QSignalBlocker(self.slider): self.slider.setValue(value)
        if not self.use_range.isChecked(): self.first.setValue(value); self.last.setValue(value)
        with QtCore.QSignalBlocker(self.table):
            if self.table.rowCount(): self.table.selectRow(value-1)
        self.preview.set_image(self.draft.render(value, self.selected), self.draft.shape)
        self.restore.setEnabled(value > 1)

    def set_state(self, state):
        a, b = (self.first.value(), self.last.value()) if self.use_range.isChecked() else (self.frame.value(),)*2
        self.change(lambda: self.draft.set_state(self.selected, a, b, state))

    def image_clicked(self, x, y):
        def act():
            frame = self.frame.value(); mode = self.mode.currentIndex()
            if mode == 1:
                self.selected = self.draft.add(frame, x, y, self.radius.value())
            elif mode == 2:
                self.draft.move(self.selected, frame, x, y)
            else:
                hits = []
                for ident, r in self.draft.records.items():
                    if self.draft.is_replaced(ident, frame): continue
                    if r['states'][frame-1] != 'present' and ident != self.selected: continue
                    px, py = r['positions'][frame-1]; distance = (x-px)**2+(y-py)**2
                    if distance <= (1.5*r['radius'])**2: hits.append((distance, ident))
                if not hits: return
                self.selected = min(hits)[1]
                if mode == 3: self.draft.set_state(self.selected, frame, frame, 'absent')
        self.change(act)


class DwellInspectionWindow(QtWidgets.QDialog):
    """A separate view of the controller's live review; never a copy of its results."""
    def __init__(self, review):
        super().__init__(review, QtCore.Qt.Window)
        self.review = review
        self.setWindowTitle('Dwell Analysis — Movie and Event Inspection')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setSizeGripEnabled(True); self.setMinimumSize(600, 460)
        screen = review.screen() or QtWidgets.QApplication.primaryScreen()
        area = screen.availableGeometry()
        self.resize(min(1440, int(area.width()*.92)), min(1050, int(area.height()*.92)))
        from window_manager import register_pyNuD_window
        register_pyNuD_window(self, 'sub')

    def restoreWindowSettings(self):
        from helperFunctions import restore_window_geometry
        import globalvals as gv
        restore_window_geometry(self, 'DwellInspectionWindow')
        settings = (getattr(gv, 'windowSettings', {}) or {}).get('DwellInspectionWindow', {})
        for name in ('splitter', 'lower_splitter', 'image_splitter', 'plot_splitter'):
            sizes = settings.get(name)
            if isinstance(sizes, list) and len(sizes) == 2 and all(isinstance(n, int) and n > 0 for n in sizes):
                getattr(self.review, name).setSizes(sizes)

    def saveWindowSettings(self):
        import globalvals as gv
        geo = self.normalGeometry() if self.isMaximized() else self.geometry()
        settings = dict(getattr(gv, 'windowSettings', {}) or {})
        settings['DwellInspectionWindow'] = dict(
            x=geo.x(), y=geo.y(), width=geo.width(), height=geo.height(), visible=False,
            title=self.windowTitle(), class_name=type(self).__name__,
            **{name: getattr(self.review, name).sizes()
               for name in ('splitter', 'lower_splitter', 'image_splitter', 'plot_splitter')})
        gv.windowSettings = settings

    def reject(self):
        self.close()

    def closeEvent(self, event):
        self.review.play.setChecked(False)
        self.saveWindowSettings()
        event.accept()


class DwellReviewDialog(QtWidgets.QDialog):
    def __init__(self, owner):
        super().__init__(owner); self.owner = owner; self.main = owner.parent_win
        self.setWindowTitle('Dwell Analysis — AI Processing')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.resize(1000, 650); self.setMinimumSize(540, 420); self.setSizeGripEnabled(True)
        from window_manager import register_pyNuD_window
        register_pyNuD_window(self, 'sub')
        self.session = None; self.worker = None; self.preparing = False; self.closing = False
        self.edit_marks_after_finish = False
        self.stop = threading.Event(); self.run_context = None; self.selected = None; self.selected_site = None; self.started = None
        self.run_budget = None; self.run_elapsed = None
        self.last_activity = None
        self.consent = AISharingConsent(self, scope='Dwell Analysis', object_name='dwell_ai_data_sharing')
        root = QtWidgets.QVBoxLayout(self)
        self.stop_notice = AIStopNotice(self, root)
        self.runtime = QtWidgets.QLabel(); self.runtime.setWordWrap(True); self.runtime.hide(); root.addWidget(self.runtime)
        recovery_row = QtWidgets.QHBoxLayout(); root.addLayout(recovery_row)
        self.resume_button = QtWidgets.QPushButton('Resume AI Processing'); self.resume_button.hide()
        self.resume_button.clicked.connect(self.resume_processing); recovery_row.addWidget(self.resume_button)
        self.resume_button.setToolTip('Retry the interrupted AI step using completed work. Resets the inactivity timer, not the total time limit.')
        self.stop_run = QtWidgets.QPushButton('Stop and review available results'); self.stop_run.hide()
        self.stop_run.clicked.connect(lambda: self.worker.ai_recovery.stop() if self.worker else None)
        recovery_row.addWidget(self.stop_run)
        self.stop_run.setToolTip('End the paused run and inspect any retained, validated measurements without further AI work.')
        self.extend_button = QtWidgets.QPushButton('Add 15 minutes'); self.extend_button.hide()
        self.extend_button.setToolTip('Extend the total processing limit by 15 minutes. Resume only resets the inactivity timer.')
        self.extend_button.clicked.connect(self.extend_time); recovery_row.addWidget(self.extend_button)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); root.addWidget(scroll)
        content = QtWidgets.QWidget(); content.setMinimumWidth(490); scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(content)
        self.source = QtWidgets.QLabel(); self.source.setWordWrap(True); layout.addWidget(self.source)
        intro = QtWidgets.QLabel('AI inspects the main-window processed movie and chooses detection and measurement methods for this data. '
            'It compares algorithms and particle scales, then checks missed/merged particles, ROIs and height traces. '
            'Drift/background measurements are reused; unresolved cases remain available for human review. '
            'Review the movie and height traces. Green: clear candidate; orange: review required; blue: unsupported / edge. '
            'A particle ROI may remain visible even when binding/unbinding is uncertain. '
            'Box/cross: particle extent/center. Circle on the selected ROI: height sampling area. '
            'Candidates are not confirmed molecular assignments. Source ASD is preserved.')
        intro.setWordWrap(True); layout.addWidget(intro)
        self.consent.add_checkbox(layout)
        self.instructions = QtWidgets.QLineEdit()
        self.instructions.setPlaceholderText('Additional instructions: target particles, suspicious intervals, or expected binding behavior.')
        self.instructions.setMinimumHeight(32); layout.addWidget(self.instructions)
        bar = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton('Start AI Analysis'); self.start.setMinimumHeight(36)
        # Native macOS default-button rendering can look disabled in this scroll area.
        # Keep availability legible even when the button/window does not have focus.
        self.start.setStyleSheet('''
            QPushButton { padding: 4px 14px; border: 2px solid transparent; border-radius: 6px; }
            QPushButton:enabled { background-color: #1769cf; color: white; }
            QPushButton:enabled:hover { background-color: #125bb5; }
            QPushButton:enabled:pressed { background-color: #104b98; }
            QPushButton:enabled:focus { border-color: #82b3ff; }
            QPushButton:disabled { background-color: #e6e8eb; color: #737b85; border-color: #d0d4da; }
        ''')
        self.start.clicked.connect(self.start_analysis); bar.addWidget(self.start)
        self.cancel_button = QtWidgets.QPushButton('Cancel Processing'); self.cancel_button.clicked.connect(self.cancel)
        bar.addWidget(self.cancel_button); bar.addWidget(QtWidgets.QLabel('Inactivity timeout'))
        self.time_limit = QtWidgets.QSpinBox(); self.time_limit.setRange(1, 60); self.time_limit.setValue(15)
        self.time_limit.setSuffix(' min'); self.time_limit.setMinimumSize(110, 36); bar.addWidget(self.time_limit)
        bar.addWidget(QtWidgets.QLabel('AI rounds'))
        self.rounds = QtWidgets.QSpinBox(); self.rounds.setRange(1, 6); self.rounds.setValue(2); self.rounds.setMinimumSize(80, 36)
        self.rounds.setToolTip('Writable AI turns per run, including the first analysis, output repairs and automatic revisions after review. Read-only reviews do not consume a round. The AI may iterate within a turn; the total time limit still applies.')
        bar.addWidget(self.rounds); bar.addStretch()
        self.time_limit.setToolTip('Pause when no new activity is observed for this long. Active AI work can continue beyond this time. '
            'The total processing limit is three times this value; Add 15 minutes extends it.')
        layout.addLayout(bar)
        retry_row = QtWidgets.QHBoxLayout()
        self.retry_review = QtWidgets.QPushButton('Retry AI Review'); self.retry_review.clicked.connect(self.retry_ai_review)
        retry_row.addWidget(self.retry_review)
        retry_help = QtWidgets.QLabel('Review retained, validated measurements without repeating detection.')
        retry_help.setWordWrap(True); retry_row.addWidget(retry_help, 1); layout.addLayout(retry_row)
        self.retry_review.setToolTip('Checks existing measurements only. Use Refine with AI to change detections or measurements.')
        self.summary = QtWidgets.QLabel('No analyzed movie yet.'); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.inspection_button = QtWidgets.QPushButton('Open Inspection Window…')
        self.inspection_button.setMinimumHeight(36); self.inspection_button.clicked.connect(self.show_inspection)
        layout.addWidget(self.inspection_button)
        self.inspection = DwellInspectionWindow(self)
        inspection_root = QtWidgets.QVBoxLayout(self.inspection)
        self.inspection_summary = QtWidgets.QLabel(); self.inspection_summary.setWordWrap(True)
        inspection_root.addWidget(self.inspection_summary)
        self.inspection_notice = QtWidgets.QLabel(); self.inspection_notice.setWordWrap(True)
        self.inspection_notice.setStyleSheet(self.stop_notice.banner.styleSheet()); self.inspection_notice.hide()
        inspection_root.addWidget(self.inspection_notice)
        self.inspection_scroll = QtWidgets.QScrollArea(); self.inspection_scroll.setWidgetResizable(True)
        inspection_root.addWidget(self.inspection_scroll, 1)
        review_content = QtWidgets.QWidget(); review_content.setMinimumWidth(900)
        self.inspection_scroll.setWidget(review_content)
        review_layout = QtWidgets.QVBoxLayout(review_content); review_layout.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical); split.setChildrenCollapsible(False); split.setHandleWidth(12)
        split.setObjectName('dwell_image_review_splitter')
        split.setStyleSheet('''
            QSplitter#dwell_image_review_splitter::handle:vertical {
                background: #bec8d4; border-top: 1px solid #96a3b2; border-bottom: 1px solid #96a3b2;
            }
            QSplitter#dwell_image_review_splitter::handle:vertical:hover { background: #8eb5e5; }
        ''')
        review_layout.addWidget(split, 1)
        movies = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(movies); ml.setContentsMargins(0, 0, 0, 0)
        images = QtWidgets.QSplitter(QtCore.Qt.Horizontal); images.setChildrenCollapsible(False); images.setHandleWidth(8)
        self.image_splitter = images
        self.original = DwellImage(); self.overlay = DwellImage()
        for title, image in [('Main-window image', self.original), ('Candidate ROIs', self.overlay)]:
            box = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(box); bl.addWidget(QtWidgets.QLabel(title)); bl.addWidget(image)
            images.addWidget(box)
        split.addWidget(images)
        # Keep the graph's useful minimum height inside a scroll area, so it
        # cannot prevent the user from giving the images most of the window.
        self.inspection_lower_scroll = QtWidgets.QScrollArea()
        self.inspection_lower_scroll.setWidgetResizable(True)
        self.inspection_lower_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.inspection_lower_scroll.setMinimumHeight(180)
        self.inspection_lower_scroll.setWidget(movies)
        split.addWidget(self.inspection_lower_scroll)
        handle = split.handle(1)
        handle.setCursor(QtCore.Qt.SplitVCursor)
        handle.setToolTip('Drag up or down to resize the images and the review area.')
        handle.setAccessibleName('Resize images and review area')
        self.overlay.clicked.connect(self.roi_clicked)
        roi_row=QtWidgets.QHBoxLayout(); roi_row.addWidget(QtWidgets.QLabel('ROI'))
        self.roi_selector=QtWidgets.QComboBox(); self.roi_selector.setMinimumWidth(150)
        self.roi_selector.currentIndexChanged.connect(self.roi_selected); roi_row.addWidget(self.roi_selector,1)
        self.roi_buttons=[]
        for text,value in [('Accept ROI','accept'),('Reject ROI','reject'),('Reset ROI','pending')]:
            button=QtWidgets.QPushButton(text); button.clicked.connect(lambda _,v=value:self.decide_roi(v))
            button.setToolTip({
                'accept': 'Confirm this particle ROI. Its Dwell events must still be accepted separately.',
                'reject': 'Exclude every event from this ROI from Apply. Measurements remain available for review.',
                'pending': 'Return this ROI to pending review. Unchanged accepted events remain eligible for Apply unless the ROI is rejected.',
            }[value])
            roi_row.addWidget(button); self.roi_buttons.append(button)
        self.correct_roi_button=QtWidgets.QPushButton('Correct ROI…'); self.correct_roi_button.clicked.connect(self.correct_roi)
        self.correct_roi_button.setToolTip('Select an existing ROI, then click one center to move it or several centers to split it.')
        roi_row.addWidget(self.correct_roi_button); ml.addLayout(roi_row)
        add_row=QtWidgets.QHBoxLayout()
        self.frame_marks_button = QtWidgets.QPushButton('Edit Frame Marks…')
        self.frame_marks_button.clicked.connect(self.edit_frame_marks); add_row.addWidget(self.frame_marks_button)
        self.add_roi_button=QtWidgets.QPushButton('Add ROI…'); self.add_roi_button.clicked.connect(self.add_roi)
        self.add_roi_button.setToolTip('Click missed particle centers to create new ROIs without changing existing ones.')
        add_row.addWidget(self.add_roi_button)
        self.show_rejected=QtWidgets.QCheckBox('Show rejected ROIs'); self.show_rejected.setChecked(True)
        self.show_rejected.toggled.connect(lambda:self.frame_changed(self.frame.value()) if self.session else None)
        add_row.addWidget(self.show_rejected); add_row.addStretch()
        self.refine_button = QtWidgets.QPushButton('Refine with AI…'); self.refine_button.setObjectName('dwell_refine')
        self.refine_button.clicked.connect(self.refine_with_ai); add_row.addWidget(self.refine_button)
        self.undo_refine_button = QtWidgets.QPushButton('Undo Review Edit')
        self.undo_refine_button.clicked.connect(self.undo_refine); add_row.addWidget(self.undo_refine_button)
        ml.addLayout(add_row)
        roi_note=QtWidgets.QLabel('Click an ROI in the right image to select it. Repeated clicks cycle overlapping ROIs. '
            'Correct ROI moves/splits the selected ROI; Add ROI creates ROIs for missed particles. '
            'Accept ROI confirms the particle only; event acceptance is separate. '
            'After Refine, an ROI may need partial re-review (pending); unchanged accepted events remain eligible for Apply. '
            'Reject ROI excludes all its events from Apply.')
        roi_note.setWordWrap(True); ml.addWidget(roi_note)
        nav = QtWidgets.QHBoxLayout(); self.play = QtWidgets.QPushButton('Play'); self.play.setCheckable(True)
        self.play.toggled.connect(self.toggle_play); nav.addWidget(self.play)
        self.previous = QtWidgets.QPushButton('−1'); self.previous.clicked.connect(lambda: self.frame.setValue(self.frame.value()-1)); nav.addWidget(self.previous)
        self.next = QtWidgets.QPushButton('+1'); self.next.clicked.connect(lambda: self.frame.setValue(self.frame.value()+1)); nav.addWidget(self.next)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(1, 1); nav.addWidget(self.slider, 1)
        self.frame = QtWidgets.QSpinBox(); self.frame.setRange(1, 1); self.frame.setMinimumWidth(90); nav.addWidget(self.frame)
        self.slider.valueChanged.connect(self.frame.setValue); self.frame.valueChanged.connect(self.frame_changed)
        ml.addLayout(nav)
        from ai_image_review import ImageReview
        self.image_review = ImageReview(self.inspection, 'Dwell Analysis', self.frame)
        nav.addWidget(self.image_review.button)
        self.image_review.picked.connect(self.pick_large_image)
        lower = QtWidgets.QSplitter(QtCore.Qt.Horizontal); lower.setChildrenCollapsible(False); lower.setHandleWidth(8)
        ml.addWidget(lower, 1)
        events_box = QtWidgets.QWidget(); el = QtWidgets.QVBoxLayout(events_box)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel('Events'))
        self.filter = QtWidgets.QComboBox(); self.filter.addItems(['All candidates', 'Clear candidates', 'Pending', 'Accepted'])
        self.filter.currentIndexChanged.connect(self.fill_table); row.addWidget(self.filter); el.addLayout(row)
        self.table = QtWidgets.QTableWidget(0, 7); self.table.setHorizontalHeaderLabels(['Event / site', 'First', 'Last', 'Dwell (s)', 'Status', 'Censored', 'Decision'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMinimumSize(260, 160); self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.event_selected); el.addWidget(self.table)
        choose = QtWidgets.QHBoxLayout()
        for label, value in [('Accept selected', 'accept'), ('Reject selected', 'reject'), ('Reset selected', 'pending')]:
            button = QtWidgets.QPushButton(label); button.clicked.connect(lambda _, v=value: self.decide(v)); choose.addWidget(button)
            button.setToolTip({
                'accept': 'Accept this event for Apply, provided its particle ROI is not rejected.',
                'reject': 'Exclude this event from Apply while retaining it for review and export.',
                'pending': 'Return this event to pending review and remove it from Apply until it is accepted again.',
            }[value])
        el.addLayout(choose); lower.addWidget(events_box)
        plots = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(plots)
        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plot_splitter.setChildrenCollapsible(False); self.plot_splitter.setHandleWidth(8)
        pl.addWidget(self.plot_splitter, 1)
        self.figure = Figure(figsize=(6, 4), tight_layout=True); self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(380, 340); self.canvas.mpl_connect('button_press_event', self.graph_clicked)
        self.plot_splitter.addWidget(self.canvas)
        self.details = QtWidgets.QLabel('Select an event to inspect its height trace.'); self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.details.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.details_scroll = QtWidgets.QScrollArea(); self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setMinimumHeight(60); self.details_scroll.setWidget(self.details)
        self.plot_splitter.addWidget(self.details_scroll)
        self.plot_splitter.setStretchFactor(0, 1); self.plot_splitter.setStretchFactor(1, 0)
        self.plot_splitter.setSizes([400, 100])
        inspect_row = QtWidgets.QHBoxLayout()
        for label, part in [('Before event', 'before'), ('Peak frame', 'peak'), ('After event', 'after')]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _, p=part: self.inspect_event(p)); inspect_row.addWidget(button)
            button.setToolTip({'before': 'Show the frame before the selected event, when available.',
                               'peak': 'Show the peak-signal frame of the selected event.',
                               'after': 'Show the frame after the selected event, when available.'}[part])
        pl.addLayout(inspect_row)
        lower.addWidget(plots); split.setSizes([500, 400]); lower.setSizes([430, 800])
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        self.splitter = split; self.lower_splitter = lower
        self.log_toggle = QtWidgets.QPushButton('Processing log / AI explanation'); self.log_toggle.setCheckable(True); layout.addWidget(self.log_toggle)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(90); self.log.setMaximumHeight(240); self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible); layout.addWidget(self.log)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); layout.addWidget(self.progress)
        self.status = QtWidgets.QLabel('Approve data sharing to start AI. Local review and export need no consent.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        controls_bottom = QtWidgets.QHBoxLayout(); root.addLayout(controls_bottom)
        bottom = QtWidgets.QGridLayout(); inspection_root.addLayout(bottom)
        self.import_button = QtWidgets.QPushButton('Import Review JSON…'); self.import_button.clicked.connect(self.import_json); controls_bottom.addWidget(self.import_button)
        self.inspection_status = QtWidgets.QLabel(); self.inspection_status.setWordWrap(True)
        bottom.addWidget(self.inspection_status, 0, 0, 1, 3)
        self.processing_button = QtWidgets.QPushButton('AI Settings / Processing Log…')
        self.processing_button.clicked.connect(self.show_processing); bottom.addWidget(self.processing_button, 1, 0)
        self.csv_button = QtWidgets.QPushButton('Export Events CSV…'); self.csv_button.clicked.connect(lambda: self.export(False)); bottom.addWidget(self.csv_button, 1, 1)
        self.json_button = QtWidgets.QPushButton('Save Review JSON…'); self.json_button.clicked.connect(lambda: self.export(True)); bottom.addWidget(self.json_button, 1, 2)
        self.apply_button = QtWidgets.QPushButton('Apply Accepted Events'); self.apply_button.clicked.connect(self.apply_result); bottom.addWidget(self.apply_button, 2, 0, 1, 2)
        self.close_inspection = QtWidgets.QPushButton('Close Inspection'); self.close_inspection.clicked.connect(self.inspection.close)
        bottom.addWidget(self.close_inspection, 2, 2)
        # Return in a frame/ROI control must not accept a particle by activating
        # the first auto-default button in this separate dialog.
        for button in self.inspection.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False); button.setDefault(False)
        self.close_button = QtWidgets.QPushButton('Close'); self.close_button.clicked.connect(self.close); controls_bottom.addWidget(self.close_button)
        self.close_button.setToolTip('Close processing and inspection windows. Running analysis is cancelled; review results are retained.')
        for control, tip in (
            (self.instructions, 'Describe target particles, suspicious time intervals or binding behavior for the next AI analysis or review.'),
            (self.cancel_button, 'Request cancellation of the current processing run.'),
            (self.inspection_button, 'Open the movie, particle ROIs, event table and height traces for review. Does not start AI.'),
            (self.roi_selector, 'Select the particle ROI to inspect. Selecting an ROI does not accept its events.'),
            (self.frame_marks_button, 'Edit particle presence and position marks in individual frames, then remeasure locally without AI. Changed events require review.'),
            (self.show_rejected, 'Show or hide rejected particle ROIs. Rejected ROIs remain excluded from Apply.'),
            (self.undo_refine_button, 'Restore the review from before the latest Refine or frame-mark edit. No AI request is sent.'),
            (self.play, 'Play or pause the review movie. Does not start analysis.'),
            (self.previous, 'Show the previous frame.'),
            (self.next, 'Show the next frame.'),
            (self.slider, 'Browse the preview frames (numbered from 1). Does not change the processing range or run AI again.'),
            (self.frame, 'Browse the preview frames (numbered from 1). Does not change the processing range or run AI again.'),
            (self.filter, 'Filter the event table by candidate or review status. Does not change acceptance or export choices.'),
            (self.table, 'Select an event to inspect its ROI and height trace. Use the decision buttons to accept or reject it.'),
            (self.canvas, 'Click a time on the height trace to inspect the corresponding movie frame.'),
            (self.log_toggle, 'Show or hide the processing log and AI explanation.'),
            (self.import_button, 'Restore a saved Dwell review JSON locally. No AI request is needed.'),
            (self.processing_button, 'Return to the Dwell AI settings, progress and log. Keeps the current review.'),
            (self.csv_button, 'Export event measurements and review decisions to CSV, including pending and rejected candidates.'),
            (self.json_button, 'Save the measured Dwell result and current review decisions to JSON for later import.'),
            (self.close_inspection, 'Close only the inspection window. The processing panel and current review remain available.'),
        ):
            control.setToolTip(tip)
        self.play_timer = QtCore.QTimer(self); self.play_timer.timeout.connect(self.advance)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.refresh_context); self.timer.start(500)
        self.consent.changed.connect(self.consent_changed); self.refresh_context()
        self.inspection.restoreWindowSettings()

    def show_inspection(self):
        if self.session is None or self.closing: return
        self.sync_inspection()
        self.inspection.show(); self.inspection.raise_(); self.inspection.activateWindow()

    def show_processing(self):
        self.show(); self.raise_(); self.activateWindow()

    def sync_inspection(self):
        source = Path(self.session['snapshots'][0].metadata.get('source_path', 'movie.asd')).name if self.session else ''
        # The full algorithm and warning text remains in Processing; it must not
        # consume the image/graph area for a verbose AI result.
        info = ''
        if self.session:
            result = self.session['result']
            info = (f" · {len(self.session['snapshots'])} frames · {self.session['frame_time_s']:g} s/frame\n"
                    f"{len(result['sites'])} ROIs · {len(result['events'])} candidate events"
                    ' · Review choices before Apply. Method and warnings: AI Settings / Processing Log…')
        self.inspection_summary.setText('Review: '+source+info)
        self.inspection_summary.setToolTip(self.summary.text())
        self.inspection_status.setText(self.status.text()+
            ('\n'+self.runtime.text() if not self.runtime.isHidden() else ''))
        self.inspection_notice.setText(self.stop_notice.banner.text())
        self.inspection_notice.setVisible(not self.stop_notice.banner.isHidden())

    def busy(self):
        return self.worker is not None or self.preparing

    def refresh_context(self):
        try: self.settings = load_provider_settings()
        except Exception: self.settings = {}
        self.source_path = current_background_source()
        self.consent.configure(self.source_path, self.settings)
        self.source.setText('Main: '+(Path(self.source_path).name if self.source_path else 'Open a height ASD file.'))
        if self.started is not None and self.run_budget is not None:
            elapsed = self.run_elapsed if self.run_elapsed is not None else time.monotonic()-self.started
            recovery = getattr(self.worker, 'ai_recovery', None)
            timing = ' / Inactivity timeout: '+AIProgress.duration(self.run_budget)
            if self.last_activity is not None and self.run_elapsed is None:
                timing += ' / Last AI activity '+AIProgress.duration(time.monotonic()-self.last_activity)+' ago'
            if recovery is not None and recovery.waiting.is_set(): timing += ' — Paused; waiting for Resume'
            budget = getattr(recovery, 'budget', None) if recovery is not None else None
            if budget is not None and self.run_elapsed is None:
                timing += ' / Total time remaining '+AIProgress.duration(budget.total_remaining())
            self.extend_button.setVisible(budget is not None and self.run_elapsed is None)
            self.runtime.setText(('Total elapsed: ' if self.run_elapsed is not None else 'Elapsed: ')+
                AIProgress.duration(elapsed)+timing)
            self.runtime.show()
        self.refresh_controls()

    def consent_changed(self):
        if self.busy() and self.run_context is not None and (not self.consent.approved or self.run_context != self.consent.context): self.cancel()
        self.refresh_controls()

    def matches_main(self):
        if not self.session or not self.source_path: return False
        snap = self.session['snapshots'][0]
        try:
            source_unchanged(snap)
            if Path(self.source_path).resolve() != Path(snap.metadata['source_path']).resolve(): return False
            import globalvals as gv
            data = getattr(gv, 'aryData_processed_1ch', None)
            if data is None: data = getattr(gv, 'aryData', None)
            return (isinstance(data, np.ndarray) and data.shape == snap.image.shape
                    and int(gv.FrameNum) == len(self.session['snapshots'])
                    and np.isclose(float(gv.FrameTime)/1000., self.session['frame_time_s'])
                    and np.allclose([gv.XScanSize, gv.YScanSize],
                                    np.array(snap.pixel_size)*np.array(snap.image.shape[::-1])))
        except (OSError, KeyError, AttributeError, ValueError): return False

    def refresh_controls(self):
        busy = self.busy(); available, reason = agent_availability(getattr(self, 'settings', {}))
        self.start.setEnabled(not busy and self.consent.approved and available)
        self.start.setToolTip(reason or 'Start a new AI analysis of the main-window processed movie to detect particles and measure candidate Dwell events.'); self.cancel_button.setEnabled(busy)
        for widget in (self.instructions, self.time_limit, self.import_button): widget.setEnabled(not busy)
        present = self.session is not None
        audit = self.session.get('audit', {}) if present else {}
        self.retry_review.setEnabled(not busy and present and available and self.consent.approved
            and self.matches_main() and audit.get('sandbox_verified', False)
            and audit.get('ai_review_status') in ('pending', 'timed_out', 'revision_requested', 'validation_failed', 'review_failed', 'provider_error'))
        self.refine_button.setEnabled(not busy and present and available and self.consent.approved and self.matches_main())
        self.refine_button.setToolTip('Send correction instructions with this result; AI can redetect particles and remeasure traces.'
            if self.consent.approved else 'Check the data-sharing consent box before sending this result and instructions to AI.')
        self.undo_refine_button.setEnabled(not busy and present and bool(audit.get('refinement_history')))
        for widget in (self.csv_button, self.json_button, self.play, self.frame, self.slider, self.previous, self.next, self.filter, self.table):
            widget.setEnabled(present and not busy)
        self.roi_selector.setEnabled(present and not busy)
        for button in self.roi_buttons: button.setEnabled(bool(self.selected_site) and not busy)
        site=self.current_site()
        editable=bool(site and site['first_frame'] <= self.frame.value() <= site['last_frame']
            and not any(a <= self.frame.value() <= b for a,b in site.get('manual_replaced_ranges',[])))
        self.correct_roi_button.setEnabled(editable and not busy)
        self.add_roi_button.setEnabled(present and not busy)
        self.frame_marks_button.setEnabled(present and not busy)
        self.apply_button.setEnabled(present and not busy and self.matches_main()
            and (not self.session['result']['events'] or
                 any(e['decision'] != 'pending' for e in self.session['result']['events']) or
                 any(s.get('review_decision') == 'reject' for s in self.session['result']['sites'])))
        self.apply_button.setToolTip('Apply explicitly accepted episodes to the matching source movie. '
            'An ROI marked pending does not block its accepted events. After Refine, only events with unchanged evidence retain acceptance. '
            'Pending/rejected events and all events in rejected ROIs are omitted.')
        self.inspection_button.setEnabled(present)
        self.sync_inspection()

    def start_analysis(self):
        self.refresh_context()
        if self.busy() or not self.start.isEnabled(): return
        self.stop.clear(); self.run_context = self.consent.context; self.preparing = True
        self.play.setChecked(False); self.refresh_controls()
        try: snapshots, dt = backend.capture_movie(self.main, self, self.stop.is_set)
        except Exception as exc: self.failed(str(exc)); return
        finally:
            self.preparing = False; self.refresh_controls()
            if self.closing: QtCore.QTimer.singleShot(0, self.close)
        self.refresh_context()
        if self.stop.is_set() or self.run_context != self.consent.context or not self.consent.approved:
            if self.closing: self.close()
            return
        settings = deepcopy(self.settings); instructions = self.instructions.text().strip(); limit = self.time_limit.value()*60
        rounds = self.rounds.value()
        self.begin_ai_run(limit)
        def job(progress):
            with tempfile.TemporaryDirectory(prefix='pynud-dwell-') as root:
                with create_agent_client(settings, root, self.stop.is_set) as client:
                    return backend.run_analysis(snapshots, dt, client, root, instructions, progress, self.stop.is_set, limit,
                                                max_rounds=rounds)
        self.launch(job, self.install_result)

    def begin_ai_run(self, limit):
        self.log.clear(); self.log_toggle.setChecked(True); self.started = time.monotonic()
        self.run_budget = limit; self.run_elapsed = None; self.run_extended = False
        self.last_activity = None
        self.stop_notice.reset(); self.progress.setFormat('%p%'); self.progress.setValue(0)

    def retry_ai_review(self):
        self.refresh_context()
        if not self.retry_review.isEnabled(): return
        self.stop.clear(); self.run_context = self.consent.context; self.play.setChecked(False)
        settings = deepcopy(self.settings); session = self.session; limit = min(180, self.time_limit.value()*60)
        self.begin_ai_run(limit)
        def job(progress):
            with tempfile.TemporaryDirectory(prefix='pynud-dwell-review-') as root:
                with create_agent_client(settings, root, self.stop.is_set) as client:
                    return backend.review_session(session, client, root, progress, self.stop.is_set, limit)
        self.launch(job, self.install_result)

    def refine_with_ai(self):
        self.refresh_context()
        if not self.refine_button.isEnabled(): return
        self.play.setChecked(False)
        session = self.session; frame = self.frame.value(); site = self.selected_site
        context = self.consent.context
        editor = DwellRefineDialog(frame, site, self.inspection)
        if editor.exec_() != QtWidgets.QDialog.Accepted:
            editor.deleteLater(); return
        instructions = editor.instructions.toPlainText().strip(); editor.deleteLater()
        self.refresh_context()
        if (not instructions or not self.refine_button.isEnabled() or session is not self.session
                or context != self.consent.context): return
        self.stop.clear(); self.run_context = context
        settings = deepcopy(self.settings); limit = self.time_limit.value()*60; rounds = self.rounds.value()
        self.begin_ai_run(limit)
        def job(progress):
            with tempfile.TemporaryDirectory(prefix='pynud-dwell-refine-') as root:
                with create_agent_client(settings, root, self.stop.is_set) as client:
                    return backend.refine_session(session, client, root, instructions, progress, self.stop.is_set,
                                                  limit, focus_frame=frame, focus_site=site, max_rounds=rounds)
        def install(revised):
            self.install_result(revised)
            if self.session is revised:
                self.frame.setValue(frame)
                self.roi_selector.setCurrentIndex(max(0, self.roi_selector.findData(site)))
        self.launch(job, install)

    def undo_refine(self):
        if not self.undo_refine_button.isEnabled(): return
        frame = self.frame.value()
        try: restored = backend.undo_refinement(self.session)
        except (ValueError, TypeError, KeyError) as exc:
            self.validation_notice('Undo is unavailable for this result: '+str(exc), popup=True); return
        self.install_result(restored)
        self.frame.setValue(frame)
        self.status.setText('Restored the result before the last Refine or frame-mark edit, including review choices. Nothing applied or saved.')

    def edit_frame_marks(self):
        if not self.frame_marks_button.isEnabled(): return
        self.play.setChecked(False); previous = self.session
        editor = FrameMarksDialog(previous, self.frame.value(), self.selected_site, self.inspection)
        if editor.exec_() != QtWidgets.QDialog.Accepted:
            editor.deleteLater(); return
        draft, frame, selected = editor.draft, editor.frame.value(), editor.selected
        editor.deleteLater()
        if previous is not self.session or self.busy() or not draft.changed: return
        self.stop.clear(); self.run_context = None; self.started = time.monotonic()
        self.run_budget = None; self.run_elapsed = None; self.runtime.hide(); self.stop_notice.reset()
        self.log.clear(); self.progress.setFormat('%p%'); self.progress.setValue(0)
        self.status.setText('Remeasuring edited frame marks locally…')
        def install(revised):
            if self.stop.is_set() or self.closing or self.session is not previous: return
            self.install_result(revised); self.frame.setValue(frame)
            self.roi_selector.setCurrentIndex(self.roi_selector.findData(selected))
            self.status.setText('Frame marks updated. Edited events are pending; review boundaries, then Accept and Apply.')
        self.launch(lambda progress: dwell_marks.commit_marks(draft, progress, self.stop.is_set), install)

    def new_manual_marks(self):
        """Capture locally for a fresh manual review; existing results are never replaced here."""
        if self.busy() or self.session is not None: return
        self.stop.clear(); self.run_context = None; self.preparing = True; self.refresh_controls()
        try:
            snapshots, dt = backend.capture_movie(self.main, self, self.stop.is_set)
            if self.stop.is_set() or self.closing: return
            self.install_result(dwell_marks.empty_session(snapshots, dt))
        except Exception as exc:
            self.failed(str(exc)); return
        finally:
            self.preparing = False; self.refresh_controls()
            if self.closing: QtCore.QTimer.singleShot(0, self.close)
        self.show_inspection(); self.edit_frame_marks()

    def launch(self, job, install):
        self.worker = DwellWorker(job, self)
        self.worker.progress.connect(self.report); self.worker.completed.connect(install)
        self.worker.failed.connect(self.failed); self.worker.finished.connect(self.finished)
        self.refresh_controls(); self.worker.start()

    def pause_for_retry(self, message):
        self.report(self.progress.value(), message); self.progress.setFormat('Paused (timeout)')
        self.stop_notice.reset(); self.stop_notice.notify(message, resume=self.resume_processing)
        self.resume_button.show(); self.stop_run.show()

    def resume_processing(self):
        if self.worker is not None:
            self.refresh_context()
            self.resume_button.setEnabled(False); self.worker.ai_recovery.resume()

    def extend_time(self):
        recovery = getattr(self.worker, 'ai_recovery', None); budget = getattr(recovery, 'budget', None)
        if budget is None: return
        before = budget.total_seconds; recovery.add_time()
        if budget.total_seconds > before: self.report(self.progress.value(), 'Added 15 minutes to the total processing limit.')
        self.refresh_context()

    def resumed(self):
        self.run_extended = True
        self.stop_notice.reset(); self.resume_button.hide(); self.resume_button.setEnabled(True); self.stop_run.hide()
        self.progress.setFormat('%p%')
        self.report(self.progress.value(), 'Resuming the interrupted step. Completed measurements are retained.')

    def report(self, percent, message):
        elapsed = int(time.monotonic()-self.started) if self.started else 0
        self.progress.setValue(percent); self.status.setText(message)
        self.log.appendPlainText(f'[{elapsed//60:02d}:{elapsed%60:02d}] {message}')

    def observed_activity(self, message, when):
        if self.stop.is_set() or self.run_elapsed is not None: return
        self.last_activity = when
        self.report(self.progress.value(), message)

    def failed(self, message):
        retained = ' Previous review results are retained.' if self.session is not None else ' No review result is available.'
        self.report(self.progress.value(), message+retained)
        if self.stop.is_set() or not backend.is_timeout(message): self.stop_notice.reset()
        if not self.stop.is_set() and self.stop_notice.notify(message+retained):
            self.progress.setFormat('Stopped (timeout)')
        elif not self.stop.is_set():
            self.progress.setFormat('Stopped (error)')
            self.validation_notice(message+retained, popup=True)

    def validation_notice(self, message, *, popup=False):
        """Keep non-timeout failures visible even when the log is off-screen."""
        self.stop_notice.reset()
        self.stop_notice.banner.setText(message); self.stop_notice.banner.show()
        if popup:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, 'Dwell Analysis — review required',
                                       message, QtWidgets.QMessageBox.Ok, self)
            box.setWindowModality(QtCore.Qt.NonModal)
            self.stop_notice.popup = box; box.show()
            QtWidgets.QApplication.alert(self.window(), 0); QtWidgets.QApplication.beep()

    def finished(self):
        self.resume_button.hide(); self.resume_button.setEnabled(True); self.stop_run.hide(); self.extend_button.hide()
        if self.run_context is not None and self.started is not None:
            self.run_elapsed = time.monotonic()-self.started
        worker = self.worker; self.worker = None; self.run_context = None
        if worker: worker.deleteLater()
        if self.started is not None:self.run_elapsed = time.monotonic()-self.started
        self.refresh_controls()
        if self.closing: self.close()
        elif self.edit_marks_after_finish and not self.stop.is_set():
            QtCore.QTimer.singleShot(0, self.edit_frame_marks)
        self.edit_marks_after_finish = False

    def cancel(self):
        self.stop.set(); self.status.setText('Cancelling… Waiting for the analysis worker to stop.')

    def install_result(self, session):
        if self.run_context is not None and (self.stop.is_set() or self.run_context != self.consent.context): return
        self.stop_notice.reset()
        self.session = session; self.selected = None; self.selected_site = None; self.owner._dwell_review_session = session
        self.fill_rois()
        count = len(session['snapshots']); self.slider.setRange(1, count); self.frame.setRange(1, count)
        self.frame.setValue(1); self.fill_table(); self.frame_changed(1); self.update_summary()
        self.status.setText('Review event boundaries and traces, accept useful events, then Apply or export. Nothing is applied automatically.')
        audit = session.get('audit', {})
        if audit.get('ai_review_status') == 'timed_out':
            message = audit['interruption']['message']
            if audit.get('sandbox_verified'):
                message += ' Use Retry AI Review to continue without repeating detection.'
            elif audit.get('refinement_request'):
                message += ' The previous result and its review decisions are retained.'
            else:
                message += ' These are local candidates, not AI-optimized results.'
            self.status.setText(message); self.progress.setFormat('Stopped (timeout)')
            self.stop_notice.notify(message, popup=self.run_context is not None and not session.get('imported'))
        elif audit.get('ai_review_status') == 'complete':
            self.stop_notice.reset(); self.progress.setFormat('%p%'); self.progress.setValue(100)
        if audit.get('ai_review_status') in ('validation_failed', 'review_failed', 'provider_error'):
            message = audit['interruption']['message']
            if not audit.get('sandbox_verified'):
                message += ' Shown results are unreviewed local candidates, not the rejected AI result.'
            else:
                message += ' Retry AI Review checks these previous validated measurements without repeating detection.'
            self.status.setText(message)
            self.progress.setFormat('Review required' if audit['ai_review_status']=='review_failed' else 'Stopped (validation error)')
            self.validation_notice(message, popup=self.run_context is not None and not session.get('imported'))
        elif audit.get('ai_review_status') == 'revision_requested':
            message = audit['interruption']['message']
            self.status.setText(message); self.progress.setFormat('Review required')
            self.validation_notice(message)
        elif session['result'].get('coverage_repairs') and audit.get('ai_review_status') != 'timed_out':
            message = (f"{len(session['result']['coverage_repairs'])} event proposals crossed unsupported frames. "
                'Only observed portions contribute to dwell times; review the censored fragments. '
                'Fragments may be parts of one event, not separate binding events. Original proposals are retained in JSON.')
            self.status.setText(message); self.validation_notice(message)
        self.refresh_controls()

        if self.isVisible() and not self.closing: self.show_inspection()

    def update_summary(self):
        if not self.session: return
        result = self.session['result']; events = result['events']
        rejected={s['id'] for s in result['sites'] if s.get('review_decision')=='reject'}
        snap = self.session['snapshots'][0]; name = Path(snap.metadata.get('source_path', 'snapshot')).name
        restored = sum(e.get('candidate_source') in ('ai_height_trace', 'local_seed') for e in events)
        event_sites = {e['site'] for e in events}
        presence_only = sum(s['id'] not in event_sites and any(s.get('particle_present', [])) for s in result['sites'])
        retention = (f'{restored} AI-omitted measured candidates retained for review (orange); none accepted automatically.\n'
                     if restored else '')
        assessment = self.session.get('audit', {}).get('method_assessment', {})
        method = (f"AI method choice: {assessment['selected_method']}\n{assessment['selection_reason']}\n"
                  if assessment else '')
        self.summary.setText(f"Review snapshot: {name} · {len(self.session['snapshots'])} frames · {self.session['frame_time_s']:g} s/frame\n"
            f"{len(result['sites'])} sites · {len(events)} events · {sum(e['status']=='clear_candidate' for e in events)} clear candidates · "
            f"{sum(e['decision']=='accept' and e['site'] not in rejected for e in events)} accepted events eligible for Apply · {len(result['assessed_frames'])} numerically assessed frames\n"
            f"ROI decisions: {sum(s.get('review_decision')=='accept' for s in result['sites'])} accepted · {len(rejected)} rejected\n"
            + (f'{presence_only} additional particle-presence sites have no detected event; their ROIs remain visible.\n' if presence_only else '')
            + method + retention + result['algorithm']+'\n'+' '.join(result['warnings']))

    def current_site(self):
        return next((s for s in self.session['result']['sites'] if s['id']==self.selected_site),None) if self.session else None

    def fill_rois(self):
        with QtCore.QSignalBlocker(self.roi_selector):
            self.roi_selector.clear(); self.roi_selector.addItem('Choose ROI…',None)
            if self.session:
                for site in self.session['result']['sites']:
                    self.roi_selector.addItem(site['id']+' · '+site.get('review_decision','pending'),site['id'])
                self.roi_selector.setCurrentIndex(max(0,self.roi_selector.findData(self.selected_site)))

    def roi_clicked(self,x,y):
        if not self.session or self.busy(): return
        hits=backend.pick_sites(self.session['result'],self.frame.value(),x,y,self.selected_site,self.show_rejected.isChecked())
        if not hits: return
        # A repeated click cycles overlapping candidates; a new location picks nearest center.
        key=(self.frame.value(),round(x,1),round(y,1),tuple(hits))
        ident=hits[(hits.index(self.selected_site)+1)%len(hits)] if getattr(self,'_last_roi_click',None)==key and self.selected_site in hits else hits[0]
        self._last_roi_click=key; self.roi_selector.setCurrentIndex(self.roi_selector.findData(ident))

    def roi_selected(self):
        if not self.session: return
        self.selected_site=self.roi_selector.currentData()
        events=[e for e in self.session['result']['events'] if e['site']==self.selected_site]
        event=next((e for e in events if e['first_frame']<=self.frame.value()<=e['last_frame']),events[0] if events else None)
        self.selected=event['id'] if event else None
        with QtCore.QSignalBlocker(self.table):
            self.table.clearSelection()
            for row in range(self.table.rowCount()):
                if self.table.item(row,0).data(QtCore.Qt.UserRole)==self.selected: self.table.selectRow(row); break
        self.frame_changed(self.frame.value()); self.refresh_controls()

    def decide_roi(self,value):
        if not self.selected_site or self.busy(): return
        backend.set_site_decision(self.session,self.selected_site,value)
        self.fill_rois(); self.fill_table(); self.update_summary(); self.frame_changed(self.frame.value()); self.refresh_controls()
        self.status.setText('ROI '+self.selected_site+': '+value+'. '+
            ('Its events are excluded from Apply; original event decisions remain in the review.' if value=='reject' else
             'Particle decision saved in this review. Binding/unbinding events still need individual review.'))

    def correct_roi(self):
        if not self.correct_roi_button.isEnabled(): return
        self.edit_rois(self.current_site())

    def add_roi(self):
        if not self.add_roi_button.isEnabled(): return
        self.edit_rois(None)

    def edit_rois(self,site):
        frame=self.frame.value(); self.play.setChecked(False)
        editor=ROICorrectionDialog(self.session,site,frame,self.inspection)
        if editor.exec_()!=QtWidgets.QDialog.Accepted:
            editor.deleteLater(); return
        centers=deepcopy(editor.centers); radius=editor.radius.value(); last=editor.last.value(); step=editor.step.value()
        editor.deleteLater()
        session=self.session; ident=site['id'] if site else None; self.stop.clear(); self.run_context=None
        self.log.clear(); self.started=time.monotonic(); self.run_elapsed=None; self.run_budget=None
        self.runtime.hide()
        self.progress.setFormat('%p%'); self.progress.setValue(0)
        self.status.setText('Remeasuring selected ROI locally…' if site else 'Measuring added ROIs locally…')
        def install(result):
            if self.stop.is_set() or self.closing: return
            self.install_result(result)
            new_id=result['audit']['roi_actions'][-1]['replacement_sites'][0]
            self.frame.setValue(frame); self.roi_selector.setCurrentIndex(self.roi_selector.findData(new_id))
            self.status.setText(('ROI correction measured. ' if site else 'New ROIs measured; existing ROIs are unchanged. ')+
                'New ROIs and events are pending; review before Apply. Edits are retained in JSON history.')
        def job(progress):
            if site: return backend.correct_roi(session,ident,frame,centers,radius,last,step,progress,self.stop.is_set)
            return backend.add_rois(session,frame,centers,radius,last,step,progress,self.stop.is_set)
        self.launch(job,install)

    def fill_table(self):
        if not self.session: return
        chosen = self.selected; kind = self.filter.currentText()
        events = [e for e in self.session['result']['events']
                  if kind == 'All candidates' or kind == 'Clear candidates' and e['status'] == 'clear_candidate'
                  or kind == 'Pending' and e['decision'] == 'pending' or kind == 'Accepted' and e['decision'] == 'accept']
        if chosen not in {e['id'] for e in events}:
            chosen = None; self.selected = None
        with QtCore.QSignalBlocker(self.table):
            self.table.setRowCount(len(events))
            for row, e in enumerate(events):
                censored = '/'.join(side for side in ('left', 'right') if e[side+'_censored']) or 'No'
                label = {'clear_candidate': 'Clear', 'review_required': 'Review', 'manual': 'Manual'}[e['status']]
                values = [e['id']+' / '+e['site'], e['first_frame'], e['last_frame'], f"{e['dwell_s']:.3g}", label, censored, e['decision']]
                if any(s['id']==e['site'] and s.get('review_decision')=='reject' for s in self.session['result']['sites']):
                    values[-1]+=' (ROI rejected)'
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value)); item.setData(QtCore.Qt.UserRole, e['id'])
                    item.setForeground(QtGui.QColor('#257655' if e['status'] == 'clear_candidate' else '#ad621a'))
                    self.table.setItem(row, column, item)
                if e['id'] == chosen: self.table.selectRow(row)
        presence_selection = self.selected_site and not any(e['site']==self.selected_site for e in self.session['result']['events'])
        if chosen is None and events and not presence_selection:
            self.table.selectRow(0)
            # Qt can retain the same selected row index after filtering, without
            # emitting itemSelectionChanged although that row now has a new ID.
            self.event_selected()
        elif not events or presence_selection:
            self.frame_changed(self.frame.value())

    def selected_event(self):
        return next((e for e in self.session['result']['events'] if e['id'] == self.selected), None) if self.session else None

    def event_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        self.selected = self.table.item(rows[0].row(), 0).data(QtCore.Qt.UserRole)
        event = self.selected_event()
        self.selected_site=event['site']
        with QtCore.QSignalBlocker(self.roi_selector):self.roi_selector.setCurrentIndex(self.roi_selector.findData(self.selected_site))
        self.frame.setValue(max(1, event['first_frame']-3)); self.frame_changed(self.frame.value())

    def decide(self, value):
        if self.busy(): return
        event = self.selected_event()
        if event is None: return
        event['decision'] = value; self.fill_table(); self.update_summary(); self.frame_changed(self.frame.value()); self.refresh_controls()

    def inspect_event(self, part):
        event = self.selected_event()
        if event is None: return
        frame = event['peak_frame'] if part == 'peak' else event['first_frame']-3 if part == 'before' else event['last_frame']+3
        self.frame.setValue(max(1, min(self.frame.maximum(), frame)))

    def frame_changed(self, value):
        with QtCore.QSignalBlocker(self.slider): self.slider.setValue(value)
        if not self.session: return
        snap = self.session['snapshots'][value-1]; event = self.selected_event()
        result, selected, rejected = self.session['result'], self.selected_site, self.show_rejected.isChecked()
        self.original.set_image(backend.render_frame(snap))
        self.overlay.set_image(backend.render_frame(snap, result, value, selected, rejected),snap.image.shape)
        def render_large():
            from ai_image_review import pil_pixmap
            edge = max(1000, *snap.image.shape)
            return [('Main-window image', pil_pixmap(backend.render_frame(snap, long_edge=edge))),
                    ('Candidate ROIs', pil_pixmap(backend.render_frame(snap, result, value, selected, rejected, long_edge=edge)))]
        self.image_review.set_images([('Main-window image', self.original.pixmap), ('Candidate ROIs', self.overlay.pixmap)],
                                     f'Frame {value}/{len(self.session["snapshots"])} · Click a candidate ROI to select it.', render=render_large)
        self.draw_plots(value)
        self.refresh_controls()

    def pick_large_image(self, index, x, y):
        if index != 1 or self.session is None: return
        h, w = self.session['snapshots'][self.frame.value()-1].image.shape
        self.roi_clicked(float(np.clip(x*w-.5, 0, w-1)), float(np.clip(h-.5-y*h, 0, h-1)))

    def draw_plots(self, frame):
        result = self.session['result']; dt = self.session['frame_time_s']
        key = (id(self.session), self.selected, self.selected_site, tuple(e['decision'] for e in result['events']),
               tuple(s.get('review_decision','pending') for s in result['sites']))
        position = (frame-1)*dt
        if getattr(self, '_plot_key', None) == key:
            for cursor in self.plot_cursors: cursor.set_xdata([position, position])
            low, high = self.trace_axis.get_xlim()
            if not low <= position <= high:
                span = max(high-low, 30*dt)
                self.trace_axis.set_xlim(max(0, position-span*.5), min(len(self.session['snapshots'])*dt, position+span*.5))
            self.canvas.draw_idle()
            return
        self._plot_key = key; self.figure.clear()
        trace = self.figure.add_subplot(211); timeline = self.figure.add_subplot(212); event = self.selected_event()
        self.trace_axis = trace; self.plot_cursors = []
        site=self.current_site()
        if site:
            frames = np.arange(site['first_frame'], site['last_frame']+1); values = np.asarray(site['response_nm'])
            valid = np.asarray(site['valid']); trace.plot((frames-1)*dt, values, color='#b4bac2', lw=.7)
            trace.plot((frames-1)*dt, np.where(valid, values, np.nan), color='#296fac', lw=1)
            if event: trace.axvspan((event['first_frame']-1)*dt, event['last_frame']*dt, color='#39bc85', alpha=.25)
            for key in ('low_nm', 'high_nm'):
                if site.get(key) is not None: trace.axhline(site[key], ls='--', color='gray', lw=.7)
            trace.set_title((event['id']+' / ' if event else '')+site['id']+' — local height trace', fontsize=12)
            lo = max(0, min((event or site)['first_frame']-15, frame-5)-1)*dt
            hi = min(len(self.session['snapshots']), max((event or site)['last_frame']+15, frame+5))*dt
            trace.set_xlim(lo, max(lo+dt, hi)); trace.set_ylabel('Height signal (nm)', fontsize=10)
        if event:
            upper = event['dwell_upper_s']; interval = f"{event['dwell_lower_s']:.3g}–{upper:.3g} s" if upper is not None else f"≥{event['dwell_lower_s']:.3g} s (censored)"
            self.details.setText(f"Frames {event['first_frame']}–{event['last_frame']} · nominal dwell {event['dwell_s']:.3g} s · sampling bounds {interval}\n"
                + site.get('signal_definition',result['signal_definition'])+'\n'
                + ('AI omitted this measured candidate. Retained for human review; event boundaries may be uncertain.\n'
                   if event.get('candidate_source') in ('ai_height_trace', 'local_seed') else '')
                + ('; '.join(event['flags']) or 'No automatic warning; human review is still required.'))
        elif site:
            self.details.setText(site['id']+': no detected event. ROI decision: '+site.get('review_decision','pending')+'\n'+site.get('signal_definition',result['signal_definition']))
        else:
            trace.text(.5, .5, 'Select an event', transform=trace.transAxes, ha='center'); self.details.setText(result['signal_definition'])
        ids = {s['id']: i for i, s in enumerate(result['sites'])}
        rejected = {s['id'] for s in result['sites'] if s.get('review_decision')=='reject'}
        for e in result['events']:
            color = '#999999' if e['decision'] == 'reject' or e['site'] in rejected else '#238c66' if e['status'] == 'clear_candidate' else '#d48823'
            timeline.plot([(e['first_frame']-1)*dt, e['last_frame']*dt], [ids[e['site']]]*2,
                          lw=4 if e['id'] == self.selected else 2, color=color)
        for ax in (trace, timeline):
            self.plot_cursors.append(ax.axvline(position, color='#e55353', lw=.8))
            ax.tick_params(labelsize=10); ax.set_xlabel('Time (s)', fontsize=10)
        timeline.set_title('All candidate episodes', fontsize=12); timeline.set_ylabel('Site', fontsize=10)
        timeline.set_xlim(0, len(self.session['snapshots'])*dt)
        if ids:
            step = max(1, int(np.ceil(len(ids)/8)))
            timeline.set_yticks(list(ids.values())[::step], list(ids)[::step], fontsize=9)
        self.canvas.draw_idle()

    def graph_clicked(self, event):
        if self.session and event.inaxes and event.xdata is not None:
            self.frame.setValue(int(round(event.xdata/self.session['frame_time_s']))+1)

    def toggle_play(self, checked):
        self.play.setText('Pause' if checked else 'Play')
        if checked and self.session: self.play_timer.start(max(50, round(self.session['frame_time_s']*1000)))
        else: self.play_timer.stop()

    def advance(self):
        if self.frame.value() == self.frame.maximum(): self.play.setChecked(False)
        else: self.frame.setValue(self.frame.value()+1)

    def export(self, as_json):
        if self.busy() or self.session is None: return
        name = Path(self.session['snapshots'][0].metadata.get('source_path', 'movie.asd')).stem
        suffix = '.dwell.json' if as_json else '.dwell.csv'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.inspection, 'Save Dwell Review', name+suffix, 'JSON (*.json)' if as_json else 'CSV (*.csv)')
        if not path: return
        session = dict(self.session, result=deepcopy(self.session['result']))
        self.run_context = None
        def job(progress):
            (backend.save_session if as_json else backend.export_csv)(session, path)
            return path
        self.launch(job, lambda p: self.status.setText('Saved '+p))

    def import_json(self, path=None):
        if self.busy(): return
        if not isinstance(path, str): path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import Dwell Review', '', 'JSON (*.json)')
        if not path: return
        self.play.setChecked(False); self.run_context = None; self.stop_notice.reset()
        self.launch(lambda progress: backend.load_session(path), self.install_result)

    def apply_result(self):
        self.refresh_context()
        if not self.apply_button.isEnabled(): return
        import globalvals as gv
        source_unchanged(self.session['snapshots'][0])
        dx, dy = self.owner._get_nm_per_display_pixel()
        size = (float(gv.XScanSize)/dx, float(gv.YScanSize)/dy)
        gv.dwell_molecules = backend.accepted_molecules(self.session, size)
        gv.dwell_proximity_results = []
        self.owner._update_info(); self.owner._refresh_display()
        self.status.setText(f'Applied {len(gv.dwell_molecules)} accepted episodes to Dwell Analysis. Manual marks and source ASD are unchanged.')

    def reject(self):
        self.close()

    def closeEvent(self, event):
        self.play.setChecked(False)
        if self.busy():
            self.closing = True; self.cancel(); event.ignore(); return
        self.closing = False; self.inspection.close(); event.accept()
