"""Particle Analysis: consent, movie identities, portable reviews and selection."""
from copy import deepcopy
from pathlib import Path
import tempfile
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from ai_data_sharing import AISharingConsent, AISharingCheckBox, current_background_source
from ai_progress import AIProgress
from ai_providers import load_provider_settings
from analysis_runtime import create_agent_client, agent_availability
from .particle_analysis_ai import (capture_frame, capture_frames, run_analysis, run_movie_analysis, render,
    make_session, selected_ids, import_session, export_session, SESSION_FORMAT,
    first_observations, set_session_selection, statistics_rows, prepare_movie_detections, normal_detection_recipe,
    select_movie_frames, complete_sampled_session, has_skipped_frames)
from .particle_analysis_ai import refine_session, undo_refinement, apply_warnings


class ParticleRefineDialog(QtWidgets.QDialog):
    def __init__(self, session, frame, particle, parent=None):
        super().__init__(parent);self.setWindowTitle('Refine Particle Analysis with AI');self.resize(660,360)
        layout=QtWidgets.QVBoxLayout(self)
        context=QtWidgets.QLabel(f'Displayed frame: {frame}'+(f' • Particle ID: {particle}' if particle is not None else ''))
        layout.addWidget(context)
        note=QtWidgets.QLabel('Describe missed particles, false detections, merged particles, boundaries or incorrect IDs. '
            'Specify source frame numbers or ranges when needed. The displayed frame is context, not a single-frame limit. '
            'Existing masks and measurements are reused; skipped frames remain skipped.')
        note.setWordWrap(True);layout.addWidget(note)
        self.instructions=QtWidgets.QPlainTextEdit();self.instructions.setObjectName('particle_refine_instructions')
        self.instructions.setPlaceholderText('Example: Split the merged particle near the center in frames 6–9, '
            'and keep the same particle IDs in later observations.')
        layout.addWidget(self.instructions,1)
        note=QtWidgets.QLabel('Revised or new particles start unchecked for review. Use Apply to update Particle Analysis. '
            'Undo Refine restores the previous result and Use choices.')
        note.setWordWrap(True);layout.addWidget(note)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        self.run_button=buttons.button(QtWidgets.QDialogButtonBox.Ok);self.run_button.setText('Refine with AI')
        self.run_button.setEnabled(False)
        self.instructions.textChanged.connect(lambda:self.run_button.setEnabled(bool(self.instructions.toPlainText().strip())))
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)


class AnalysisWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, job, parent):
        super().__init__(parent); self.job = job
        from ai_recovery import attach_recovery
        attach_recovery(self, parent)

    def run(self):
        try: self.completed.emit(self.job(self.progress.emit))
        except Exception as exc: self.failed.emit(str(exc) or 'Analysis cancelled. Previous result retained.')


class ParticlePreview(QtWidgets.QWidget):
    picked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent); self.pixmap=None; self.labels=None; self.image_rect=QtCore.QRect()
        self.setMinimumSize(240,180)
        self.setToolTip('Click a particle to select its row. Use the table checkbox to include or exclude it.')

    def set_image(self, image, labels=None):
        rgb=image.convert('RGB'); data=rgb.tobytes()
        q=QtGui.QImage(data,rgb.width,rgb.height,rgb.width*3,QtGui.QImage.Format_RGB888).copy()
        self.pixmap=QtGui.QPixmap.fromImage(q);self.labels=labels;self.update()

    def paintEvent(self,event):
        painter=QtGui.QPainter(self);painter.fillRect(self.contentsRect(),QtGui.QColor('#121820'))
        if self.pixmap is None:
            painter.setPen(QtGui.QColor('#c3cdd8'))
            painter.drawText(self.contentsRect(),QtCore.Qt.AlignCenter,'Start AI Analysis to inspect this frame.');return
        size=self.pixmap.size();size.scale(self.size(),QtCore.Qt.KeepAspectRatio)
        self.image_rect=QtCore.QRect((self.width()-size.width())//2,(self.height()-size.height())//2,size.width(),size.height())
        painter.drawPixmap(self.image_rect,self.pixmap)

    def mousePressEvent(self,event):
        if self.labels is None or not self.image_rect.contains(event.pos()):return
        h,w=self.labels.shape
        x=min(w-1,int((event.x()-self.image_rect.x())*w/max(1,self.image_rect.width())))
        y=h-1-min(h-1,int((event.y()-self.image_rect.y())*h/max(1,self.image_rect.height())))
        ident=int(self.labels[y,x])
        if ident:self.picked.emit(ident)


COLUMNS=[('id','ID'),('first_frame_1based','First frame'),('status','Status'),('detection','Detection'),('area_nm2','Area (nm²)'),('equivalent_diameter_nm','Eq. diameter (nm)'),
    ('perimeter_nm','Perimeter (nm)'),('circularity','Circularity'),('major_axis_nm','Major axis (nm)'),
    ('minor_axis_nm','Minor axis (nm)'),('aspect_ratio','Aspect ratio'),('orientation_deg','Angle (°)'),
    ('max_z_nm','Peak Z (nm)'),('mean_z_nm','Mean Z (nm)'),('std_z_nm','Z SD (nm)'),
    ('background_nm','Local bg (nm)'),('max_above_bg_nm','Max Δbg (nm)'),('mean_above_bg_nm','Mean Δbg (nm)'),
    ('volume_above_bg_nm3','Volume Δbg (nm³)'),('x_nm','X (nm)'),('y_nm','Y (nm)')]

DETECTION_APPROACHES={
    'default_detection':'Default detection + AI review / correction',
    'ai_from_scratch':'AI designs detection from scratch',
}


class _ReviewSection(QtWidgets.QWidget):
    """A splitter does not otherwise reserve height for wrapped result labels."""
    def event(self,event):
        result=super().event(event)
        if event.type() in (QtCore.QEvent.Resize,QtCore.QEvent.LayoutRequest) and self.layout() is not None:
            height=self.layout().totalHeightForWidth(self.width())
            if height>0 and height!=self.minimumHeight():self.setMinimumHeight(height)
        return result


class ParticleAnalysisDialog(QtWidgets.QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner=owner;self.worker=None;self.result=None;self.selected=set();self.highlight=None
        self.session=None;self.preparing=False;self.worker_mode=None
        self.stop=threading.Event();self.closing=False;self.run_context=None
        self.setWindowTitle('AI Particle Analysis')
        self.resize(1160,900)
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self,'sub')
        except ImportError:pass
        outer=QtWidgets.QVBoxLayout(self)
        self.scroll_area=QtWidgets.QScrollArea();self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Ignored)
        outer.addWidget(self.scroll_area)
        content=QtWidgets.QWidget();content.setMinimumWidth(800)
        content_layout=QtWidgets.QVBoxLayout(content);content_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.scroll_area.setWidget(content)
        self.section_splitter=QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.section_splitter.setChildrenCollapsible(False)
        content_layout.addWidget(self.section_splitter,1)
        settings_section=_ReviewSection();layout=QtWidgets.QVBoxLayout(settings_section)
        settings_section.setMinimumHeight(430)
        self.source=QtWidgets.QLabel();self.source.setWordWrap(True);layout.addWidget(self.source)
        self.info=QtWidgets.QLabel();self.info.setWordWrap(True);layout.addWidget(self.info)
        self.consent=AISharingConsent(self,scope='Particle Analysis',object_name='particle_analysis_consent')
        self.consent_box=AISharingCheckBox(self.consent);layout.addWidget(self.consent_box)
        sharing=QtWidgets.QLabel('Sends the captured numerical height image, previews, calibration and your instructions. '
            'Applies to Particle Analysis for this file in this session. Changing the file or AI settings '
            'clears consent. Local review and CSV / JSON export do not require it.')
        sharing.setWordWrap(True);layout.addWidget(sharing)
        self.instructions=QtWidgets.QLineEdit();self.instructions.setPlaceholderText(
            'Additional instructions: describe the particles, weak features, touching particles, or structures to exclude.');layout.addWidget(self.instructions)
        self.scope_label=QtWidgets.QLabel()
        self.scope_label.setToolTip('Select AI frames in the Particle Analysis main panel before starting.')
        layout.addWidget(self.scope_label)
        approach_row=QtWidgets.QHBoxLayout()
        approach_label=QtWidgets.QLabel('Detection approach');approach_row.addWidget(approach_label)
        self.detection_approach=QtWidgets.QComboBox()
        for key,label in DETECTION_APPROACHES.items():self.detection_approach.addItem(label,key)
        self.detection_approach.setMinimumHeight(36)
        approach_label.setBuddy(self.detection_approach);approach_row.addWidget(self.detection_approach,1)
        layout.addLayout(approach_row)
        self.approach_help=QtWidgets.QLabel();self.approach_help.setWordWrap(True);layout.addWidget(self.approach_help)
        controls=QtWidgets.QHBoxLayout();controls.setSpacing(12)
        time_label=QtWidgets.QLabel('Inactivity timeout');controls.addWidget(time_label)
        self.time_limit=QtWidgets.QSpinBox();self.time_limit.setRange(1,60);self.time_limit.setValue(10);self.time_limit.setSuffix(' min')
        self.time_limit.setFont(time_label.font())
        control_height=max(36,self.time_limit.fontMetrics().height()+16)
        self.time_limit.setMinimumSize(max(120,self.time_limit.fontMetrics().horizontalAdvance('60 min')+56),control_height)
        self.time_limit.setSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed)
        self.time_limit.setStyleSheet('QSpinBox { padding: 4px 28px 4px 8px; } '
            'QSpinBox::up-button, QSpinBox::down-button { width: 24px; }')
        self.time_limit.setToolTip('Pause after 1–60 minutes without new activity. Ongoing AI work resets this waiting time. '
            'The total processing limit is three times this value; Add 15 minutes extends it.')
        time_label.setBuddy(self.time_limit);controls.addWidget(self.time_limit)
        rounds_label=QtWidgets.QLabel('AI rounds');controls.addWidget(rounds_label)
        self.rounds=QtWidgets.QSpinBox();self.rounds.setRange(1,6);self.rounds.setValue(2);self.rounds.setFont(time_label.font())
        self.rounds.setMinimumSize(max(90,self.rounds.fontMetrics().horizontalAdvance('66')+56),control_height)
        self.rounds.setSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed)
        self.rounds.setStyleSheet(self.time_limit.styleSheet())
        self.rounds.setToolTip('Writable AI turns per run: the first analysis plus automatic revisions after each read-only review. The AI iterates freely inside a turn; this bounds the host-mediated review cycles. The total time limit still applies.')
        rounds_label.setBuddy(self.rounds);controls.addWidget(self.rounds)
        self.start_button=QtWidgets.QPushButton('Start AI Analysis');self.start_button.clicked.connect(self.start);controls.addWidget(self.start_button)
        self.cancel_button=QtWidgets.QPushButton('Cancel Processing');self.cancel_button.clicked.connect(self.cancel);controls.addWidget(self.cancel_button)
        for button in (self.start_button,self.cancel_button):button.setMinimumHeight(control_height)
        controls.addStretch()
        layout.addLayout(controls)
        self.skip_overlap=QtWidgets.QCheckBox('Skip analysis of highly overlapping frames (≥90%, All Frames)')
        self.skip_overlap.setChecked(True)
        self.skip_overlap.setToolTip('Compare with the last analyzed representative, not just the previous frame. '
            'Skip detection, measurements and AI submission when field overlap and image agreement are high. '
            'Poor image agreement triggers analysis. Small changes and new particles can be missed in skipped frames.')
        layout.addWidget(self.skip_overlap)
        skip_note=QtWidgets.QLabel('Skipped frames show the image only, with no IDs or measurements. '
            'Turn off to analyze every frame; statistics use the first analyzed appearance.')
        skip_note.setWordWrap(True);layout.addWidget(skip_note)
        self.reuse_overlap=QtWidgets.QCheckBox('Reuse detections in stable overlapping views (≥90%, All Frames)')
        self.reuse_overlap.setChecked(True)
        self.reuse_overlap.setToolTip('Check each frame against a fully detected reference. Reuse unchanged contours; '
            'detect newly visible or changed areas. Uncertain registration falls back to full detection. '
            'AI still verifies identities; statistics use first appearances only. Used with default detection when Skip analysis is off.')
        layout.addWidget(self.reuse_overlap)
        self.skip_overlap.toggled.connect(self.refresh_controls)
        layout.addStretch(1);self.section_splitter.addWidget(settings_section)
        review_section=_ReviewSection();layout=QtWidgets.QVBoxLayout(review_section)
        self.result_label=QtWidgets.QLabel('No analyzed frame yet.');self.result_label.setWordWrap(True);layout.addWidget(self.result_label)
        options=QtWidgets.QHBoxLayout()
        self.include_uncertain=QtWidgets.QCheckBox('Include purple (uncertain)')
        self.include_edge=QtWidgets.QCheckBox('Include blue (edge / partial)')
        for box in (self.include_uncertain,self.include_edge):
            box.setToolTip('Bulk selection resets row selections. A particle with both flags needs both options. Individual Use checkboxes override the bulk selection.')
            box.toggled.connect(self.bulk_selection);options.addWidget(box)
        self.show_ids=QtWidgets.QCheckBox('Show IDs');self.show_ids.setChecked(True);self.show_ids.toggled.connect(self.update_preview);options.addWidget(self.show_ids)
        self.original=QtWidgets.QCheckBox('Show original');self.original.toggled.connect(self.update_preview);options.addWidget(self.original);options.addStretch()
        self.enlarge_button=QtWidgets.QPushButton('Open Image Review…');options.addWidget(self.enlarge_button)
        layout.addLayout(options)
        refine_row=QtWidgets.QHBoxLayout()
        self.refine_button=QtWidgets.QPushButton('Refine with AI…');self.refine_button.setObjectName('particle_analysis_refine')
        self.refine_button.clicked.connect(self.refine_with_ai);refine_row.addWidget(self.refine_button)
        self.undo_refine_button=QtWidgets.QPushButton('Undo Refine');self.undo_refine_button.clicked.connect(self.undo_refine)
        refine_row.addWidget(self.undo_refine_button)
        note=QtWidgets.QLabel('Revise the current result with instructions; review before Apply.')
        note.setWordWrap(True);refine_row.addWidget(note,1);layout.addLayout(refine_row)
        self.navigation=QtWidgets.QWidget();nav=QtWidgets.QHBoxLayout(self.navigation);nav.setContentsMargins(0,0,0,0)
        nav.addWidget(QtWidgets.QLabel('Frame'))
        self.frame_slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);nav.addWidget(self.frame_slider,1)
        self.frame_spin=QtWidgets.QSpinBox();nav.addWidget(self.frame_spin)
        from ai_image_review import ImageReview
        self.image_review=ImageReview(self,'Particle Analysis',self.frame_spin,self.enlarge_button)
        self.image_review.picked.connect(self.pick_large_image)
        self.all_flags_button=QtWidgets.QPushButton('Apply flags to all frames');self.all_flags_button.clicked.connect(self.apply_all_flags);nav.addWidget(self.all_flags_button)
        self.frame_slider.valueChanged.connect(self.switch_frame);self.frame_spin.valueChanged.connect(self.seek_frame)
        self.navigation.hide();layout.addWidget(self.navigation)
        split=self.result_splitter=QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.setChildrenCollapsible(False);split.setMinimumHeight(324)
        self.preview=ParticlePreview();self.preview.picked.connect(self.pick);split.addWidget(self.preview)
        self.table=QtWidgets.QTableWidget(0,len(COLUMNS)+1);self.table.setHorizontalHeaderLabels(['Use']+[c[1] for c in COLUMNS])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows);self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers);self.table.itemChanged.connect(self.selection_changed)
        self.table.itemSelectionChanged.connect(self.row_selected);self.table.setMinimumHeight(140);split.addWidget(self.table)
        split.setMinimumHeight(330)
        split.setSizes([490,190]);layout.addWidget(split,1)
        self.counts=QtWidgets.QLabel();self.counts.setWordWrap(True);layout.addWidget(self.counts)
        self.details=QtWidgets.QLabel();self.details.setWordWrap(True);self.details.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse);layout.addWidget(self.details)
        self.section_splitter.addWidget(review_section)
        log_section=QtWidgets.QWidget();layout=QtWidgets.QVBoxLayout(log_section)
        self.info_toggle=QtWidgets.QToolButton();self.info_toggle.setText('AI explanation / Processing log')
        self.info_toggle.setCheckable(True);self.info_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.info_toggle.setArrowType(QtCore.Qt.RightArrow);layout.addWidget(self.info_toggle)
        self.summary=QtWidgets.QPlainTextEdit();self.summary.setReadOnly(True);self.summary.setMinimumHeight(60);self.summary.hide();layout.addWidget(self.summary)
        self.bar=QtWidgets.QProgressBar();layout.addWidget(self.bar)
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.activity=AIProgress(self.bar,self.status,layout,self,prefix='particle_analysis')
        self.log_splitter=QtWidgets.QSplitter(QtCore.Qt.Vertical);self.log_splitter.setChildrenCollapsible(False)
        layout.removeWidget(self.summary);layout.removeWidget(self.activity.log)
        self.activity.log.setMaximumHeight(16777215);self.activity.log.setMinimumHeight(80)
        self.log_splitter.addWidget(self.summary);self.log_splitter.addWidget(self.activity.log)
        layout.insertWidget(1,self.log_splitter,1);self.log_splitter.hide()
        self.section_splitter.addWidget(log_section);self.section_splitter.setSizes([430,540,120])
        for splitter in (self.section_splitter,self.result_splitter,self.log_splitter):
            splitter.setHandleWidth(8)
            splitter.setStyleSheet('QSplitter::handle { background: #b9bec6; border: 1px solid #969da8; border-radius: 3px; }')
            for index in range(1,splitter.count()):splitter.handle(index).setToolTip('Drag to resize the adjacent areas')
        self.info_toggle.toggled.connect(self.toggle_activity)
        bottom=QtWidgets.QHBoxLayout()
        bottom.addStretch()
        self.apply_button=QtWidgets.QPushButton('Apply to Particle Analysis')
        self.apply_button.setToolTip('Copy the reviewed session, IDs and particle choices to the Particle Analysis main panel.')
        self.apply_button.clicked.connect(self.apply_to_main);bottom.addWidget(self.apply_button)
        self.csv_button=QtWidgets.QPushButton('Export Statistics CSV…');self.csv_button.clicked.connect(lambda:self.export('.csv'));bottom.addWidget(self.csv_button)
        self.json_button=QtWidgets.QPushButton('Save Analysis JSON…');self.json_button.clicked.connect(lambda:self.export('.json'));bottom.addWidget(self.json_button)
        self.close_button=QtWidgets.QPushButton('Close');self.close_button.clicked.connect(self.close);bottom.addWidget(self.close_button);content_layout.addLayout(bottom)
        self.setMinimumSize(480,360);self.setSizeGripEnabled(True)
        for control, tip in (
            (self.instructions, 'Describe the target particles, touching particles, weak features or structures to exclude. Sent with the next AI analysis.'),
            (self.detection_approach, 'Choose the default detection workflow with AI review, or let AI design detection for this data.'),
            (self.start_button, 'Analyze the frames selected in the Particle Analysis main panel using the main-window processed images.'),
            (self.cancel_button, 'Request cancellation of the current processing run.'),
            (self.show_ids, 'Show or hide particle ID labels. Does not change particle selections or measurements.'),
            (self.original, 'Show the captured input image without particle overlays. This does not remove the main-window processing already captured in the input.'),
            (self.undo_refine_button, 'Restore the analysis result from before the latest refinement without sending another AI request.'),
            (self.frame_slider, 'Browse the preview frames (numbered from 1). Does not change the processing range or run AI again.'),
            (self.frame_spin, 'Browse the preview frames (numbered from 1). Does not change the processing range or run AI again.'),
            (self.all_flags_button, 'Reset particle Use choices in all analyzed frames using the Include uncertain and Include edge options. Does not apply results to the main panel.'),
            (self.table, 'Select a row to highlight a particle. The Use checkbox determines whether it contributes to the reviewed statistics.'),
            (self.info_toggle, 'Show or hide the AI explanation and processing log.'),
            (self.csv_button, 'Export the reviewed particle statistics to CSV using the current Use choices.'),
            (self.json_button, 'Save the analysis, measurements and review choices as JSON for later inspection.'),
            (self.close_button, 'Close this preview. This does not undo results already applied or saved.'),
        ):
            control.setToolTip(tip)
        self.detection_approach.currentIndexChanged.connect(self.refresh_controls)
        self.consent.changed.connect(self.consent_changed)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh_context);self.timer.start(500)
        self.refresh_context()

    def refresh_context(self):
        import globalvals as gv
        self.source_path=current_background_source()
        try:self.settings=load_provider_settings()
        except Exception:self.settings={}
        self.consent.configure(self.source_path,self.settings)
        name=Path(self.source_path).name if self.source_path else 'Open a height ASD in the main window.'
        self.source.setText(f'Main: {name} • Frame {int(getattr(gv,"index",0))+1} • Channel 1')
        self.refresh_controls()

    def consent_changed(self):
        if (self.worker_mode in ('analysis','refine') or self.preparing) and (not self.consent.approved or self.consent.context!=self.run_context):
            self.cancel()
        self.refresh_controls()

    def analysis_scope(self):
        selector=getattr(self.owner,'ai_frame_scope',None)
        return 'all' if selector is not None and selector.currentData()=='all' else 'current'

    def matches_review_source(self):
        if not self.session or not self.source_path:return False
        try:
            path=Path(self.source_path).resolve();stat=path.stat()
            return all(Path(r['snapshot'].metadata.get('source_path','')).resolve()==path
                and r['snapshot'].metadata.get('source_size')==stat.st_size
                and r['snapshot'].metadata.get('source_mtime_ns')==stat.st_mtime_ns for r in self.session['frames'])
        except OSError:return False

    def refresh_controls(self):
        available,reason=agent_availability(getattr(self,'settings',{}))
        busy=self.worker is not None or self.preparing
        use_default=self.detection_approach.currentData()=='default_detection'
        self.detection_approach.setEnabled(not busy)
        self.approach_help.setText(
            'Faster starting point: reuse matching main-panel detections, or run the normal detector with the current settings. '
            'AI reviews and corrects them; it can change the method where the data require it.' if use_default else
            'AI inspects the images and chooses, tests and implements a detection method for these data. '
            'Normal detections are not supplied. This can take longer; overlap-based frame skipping still applies when enabled.')
        self.skip_overlap.setEnabled(not busy and self.analysis_scope()=='all')
        self.reuse_overlap.setEnabled(not busy and use_default and self.analysis_scope()=='all' and not self.skip_overlap.isChecked())
        selector=getattr(self.owner,'ai_frame_scope',None)
        if selector is not None:selector.setEnabled(not busy)
        scope='All Frames' if self.analysis_scope()=='all' else 'Single Frame (current main frame)'
        self.scope_label.setText('Frames: '+scope)
        self.info.setText('Uses the main-window processed height image (channel 1). '+
            (('Normal detection supplies contours and a local ID proposal. AI reviews it first and repairs only identified problems. '
             'Highly overlapping views can be skipped before detection and AI analysis. ' if self.analysis_scope()=='all' else
             'AI reviews normal detection first; correct contours are reused and only defects are repaired. pyNuD measures. ') if use_default else
             'AI chooses the detection method from the images; pyNuD validates contours and measures particles. '+
             ('AI also links movie-global particle IDs. ' if self.analysis_scope()=='all' else ''))+
            'Green: stable. Purple: uncertain. Blue: edge / partial.')
        self.start_button.setEnabled(bool(self.source_path and self.consent.approved and available and not busy))
        self.start_button.setToolTip(reason or 'Analyze the selected frames using the main-window processed images. Review particle choices before Apply.')
        self.cancel_button.setEnabled(busy and self.worker_mode!='export')
        self.instructions.setEnabled(not busy);self.time_limit.setEnabled(not busy)
        self.csv_button.setEnabled(bool(self.session and statistics_rows(self.session) and not busy))
        self.json_button.setEnabled(bool(self.result and not busy))
        self.apply_button.setEnabled(bool(self.session and not busy and callable(getattr(self.owner,'applyAnalysisSession',None))))
        self.enlarge_button.setEnabled(bool(self.result and not busy))
        refinable=bool(self.result and self.result.get('origin')!='skipped' and self.session and
            all(not p.get('geometry_kind') for r in self.session['frames'] for p in r['particles']))
        self.refine_button.setEnabled(refinable and not busy and available and self.consent.approved and self.matches_review_source())
        self.refine_button.setToolTip('Revise this captured result using AI; open the same unchanged source file and approve data sharing. '
            'Select an analyzed frame. Ring-only contour results are not segmentation masks.')
        self.undo_refine_button.setEnabled(bool(self.session and self.session.get('refinement_history') and not busy))
        self.table.setEnabled(not busy)
        self.navigation.setEnabled(not busy)
        for box in (self.include_uncertain,self.include_edge):box.setEnabled(bool(self.result and not busy))
        if not busy and not self.result and self.activity.started is None:
            self.status.setText(reason if not available else 'Approve data sharing, then Start AI Analysis.')

    def start(self):
        self.refresh_context()
        if not self.start_button.isEnabled():return
        self.run_context=self.consent.context;self.stop.clear();self.closing=False
        use_default=self.detection_approach.currentData()=='default_detection'
        approach=DETECTION_APPROACHES[self.detection_approach.currentData()]
        self.activity.start('Capturing main-window images…');self.info_toggle.setChecked(True)
        self.preparing=True;self.refresh_controls()
        try:
            if callable(getattr(self.owner.main,'stopPlayback',None)):self.owner.main.stopPlayback()
            all_frames=self.analysis_scope()=='all'
            snapshots=capture_frames(self.owner.main,self,self.stop.is_set) if all_frames else [capture_frame(self.owner.main)]
            # Freeze widget settings and cached masks on the GUI thread. Detection
            # and ID linking use only these copies in the worker.
            recipe=None;cache={}
            if use_default:
                recipe=normal_detection_recipe(self.owner.normalDetectionSettings() if
                    callable(getattr(self.owner,'normalDetectionSettings',None)) else None)
                cache={i:{**deepcopy({k:v for k,v in r.items() if k!='snapshot'}),'snapshot':r['snapshot']}
                    for i,r in getattr(self.owner,'_manual_analysis_frames',{}).items()
                    if i in {s.metadata['frame_0based'] for s in snapshots}}
        except Exception as exc:
            self.failed(str(exc));return
        finally:
            self.preparing=False;self.refresh_controls()
            if self.closing:self.close()
        self.refresh_context()
        if self.stop.is_set() or self.closing or not self.consent.approved or self.run_context!=self.consent.context:return
        provider=deepcopy(self.settings);instructions=self.instructions.text().strip();time_limit=self.time_limit.value()*60
        max_rounds=self.rounds.value()
        self.activity.start(f'Captured {len(snapshots)} frame(s). {approach}. Preparing the analysis agent…')
        self.info_toggle.setChecked(True);self.summary.hide();self.preview.set_image(render(snapshots[0]))
        skip_overlap=all_frames and self.skip_overlap.isChecked()
        reuse_overlap=use_default and all_frames and self.reuse_overlap.isChecked() and not skip_overlap
        def job(progress):
            with tempfile.TemporaryDirectory(prefix='pynud-particle-analysis-') as root:
                plan=select_movie_frames(snapshots,progress,self.stop.is_set) if skip_overlap else None
                retained=[s for s,p in zip(snapshots,plan) if p['status']=='analyzed'] if plan is not None else snapshots
                seeds=prepare_movie_detections(retained,recipe,cache,progress,self.stop.is_set,reuse_overlap=reuse_overlap) if use_default else None
                with create_agent_client(provider,root,self.stop.is_set) as client:
                    if all_frames:
                        session=run_movie_analysis(retained,client,root,instructions,progress,self.stop.is_set,
                            time_limit=time_limit,max_rounds=max_rounds,initial_detections=seeds,frame_sampling=plan)
                        return complete_sampled_session(snapshots,session,plan,self.stop.is_set) if plan is not None else session
                    return run_analysis(snapshots[0],client,root,instructions,progress,self.stop.is_set,
                        time_limit=time_limit,max_rounds=max_rounds,initial_detection=seeds[0] if seeds is not None else None)
        self.launch_worker(job,self.install_result,'analysis')

    def launch_worker(self,job,install,mode):
        self.worker_mode=mode;self.worker=AnalysisWorker(job,self)
        self.worker.progress.connect(self.activity.update);self.worker.completed.connect(install)
        self.worker.failed.connect(self.failed);self.worker.finished.connect(self.finished)
        self.refresh_controls();self.worker.start()

    def refine_with_ai(self):
        self.refresh_context()
        if not self.refine_button.isEnabled():return
        session=self.session;context=self.consent.context
        frame=self.result['snapshot'].metadata['frame_0based']+1;particle=self.highlight
        editor=ParticleRefineDialog(session,frame,particle,self)
        try:
            if editor.exec_()!=QtWidgets.QDialog.Accepted:return
            instructions=editor.instructions.toPlainText().strip()
        finally:editor.deleteLater()
        self.refresh_context()
        if (not instructions or self.session is not session or not self.refine_button.isEnabled()
                or self.consent.context!=context):return
        self.run_context=context;self.stop.clear();self.closing=False
        provider=deepcopy(self.settings);time_limit=self.time_limit.value()*60;max_rounds=self.rounds.value()
        self.activity.start('Refining the current result; previous result retained until completion…')
        self.info_toggle.setChecked(True)
        def job(progress):
            with tempfile.TemporaryDirectory(prefix='pynud-particle-refine-') as root:
                with create_agent_client(provider,root,self.stop.is_set) as client:
                    return refine_session(session,client,root,instructions,progress,self.stop.is_set,
                        time_limit=time_limit,focus_frame=frame,focus_id=particle,max_rounds=max_rounds)
        def install(revised):
            self.refresh_context()
            if self.session is not session or not self.matches_review_source():
                self.failed('Source or review changed. Previous result retained.');return
            self.install_result(revised)
            if self.session is revised:
                self.seek_frame(frame)
                incomplete=any(r.get('audit',{}).get('ai_review_status') in ('incomplete','timed_out') for r in revised['frames'])
                if not incomplete:  # install_result already reports an incomplete review.
                    self.activity.finish('Refined result ready. Revised/new particles are unchecked: review their Use choices, '
                        'then Apply to Particle Analysis. Undo Refine restores the previous result.')
        self.launch_worker(job,install,'refine')

    def undo_refine(self):
        if not self.undo_refine_button.isEnabled():return
        session=self.session;frame=self.result['snapshot'].metadata['frame_0based']+1
        self.stop.clear();self.activity.start('Restoring the result before Refine…')
        def install(restored):
            if self.stop.is_set() or self.closing or self.session is not session:
                self.failed('Restore cancelled. Current result retained.');return
            self.show_session(restored);self.seek_frame(frame)
            self.activity.finish('Restored the result before Refine, including Use choices. Apply again to update Particle Analysis.')
        self.launch_worker(lambda progress:undo_refinement(session,self.stop.is_set),install,'undo')

    def import_json(self, path=None):
        if self.worker is not None or self.preparing:return
        if not isinstance(path,str):
            path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Import Particle Analysis JSON','','JSON (*.json)')
        if not path:return
        self.stop.clear();self.closing=False
        self.activity.start('Importing saved segmentation and IDs locally…');self.info_toggle.setChecked(True)
        def install(session):
            if self.stop.is_set() or self.closing:
                self.failed('Import cancelled. Previous result retained.');return
            self.show_session(session)
            self.activity.finish('Imported segmentation and IDs. No AI request was made.');self.info_toggle.setChecked(False)
        self.launch_worker(lambda progress:import_session(path,progress,self.stop.is_set),install,'import')

    def configure_navigation(self):
        frames=self.session['frames'];self.navigation.setVisible(len(frames)>1)
        self.all_flags_button.setVisible(self.session['id_scope']!='movie-global')
        with QtCore.QSignalBlocker(self.frame_slider):self.frame_slider.setRange(0,len(frames)-1)
        with QtCore.QSignalBlocker(self.frame_spin):
            self.frame_spin.setRange(frames[0]['snapshot'].metadata['frame_0based']+1,frames[-1]['snapshot'].metadata['frame_0based']+1)

    def show_session(self,session):
        self.session=session;self.configure_navigation();self.switch_frame(0)

    def switch_frame(self,index):
        if not self.session or not 0<=index<len(self.session['frames']):return
        result=self.session['frames'][index]
        with QtCore.QSignalBlocker(self.frame_slider):self.frame_slider.setValue(index)
        with QtCore.QSignalBlocker(self.frame_spin):self.frame_spin.setValue(result['snapshot'].metadata['frame_0based']+1)
        self.show_result(result,keep_session=True)

    def seek_frame(self,value):
        if not self.session:return
        index=min(range(len(self.session['frames'])),key=lambda i:abs(self.session['frames'][i]['snapshot'].metadata['frame_0based']+1-value))
        self.switch_frame(index)

    def apply_all_flags(self):
        if not self.session:return
        for result in self.session['frames']:
            result['selected_ids']={p['id'] for p in result['particles'] if
                (not p['uncertain'] or self.include_uncertain.isChecked()) and (not p['edge'] or self.include_edge.isChecked())}
        self.show_result(self.result,keep_session=True)

    def failed(self,message):
        self.activity.fail(message)
        if self.result:self.update_preview()

    def install_result(self,result):
        self.refresh_context()
        if self.stop.is_set() or self.closing or not self.consent.approved or self.consent.context!=self.run_context:
            self.failed('Analysis cancelled or source/AI destination changed. Previous result retained.');return
        if result.get('format')==SESSION_FORMAT:self.show_session(result)
        else:self.show_result(result)
        skipped=sum(r.get('origin')=='skipped' for r in self.session['frames'])
        audit=next((r.get('audit',{}) for r in self.session['frames'] if r.get('origin')!='skipped'),{})
        route=('AI accepted the local ID proposal (1 review; no analysis code). '
               if audit.get('execution_mode')=='local-id-proposal-review' else '')
        self.activity.finish((f'{len(self.session["frames"])-skipped} frames analyzed; {skipped} skipped. ' if skipped else '')+
            route+'Review particle choices. Statistics CSV uses one first-analyzed-appearance measurement per movie ID; JSON saves the complete review.')
        incomplete=any(r.get('audit',{}).get('ai_review_status') in ('incomplete','timed_out') for r in self.session['frames'])
        if incomplete:
            message='AI review incomplete. Measured contours and IDs are retained; check the flagged particles before Apply.'
            self.activity.finish(message)
            self.activity.bar.setFormat('Review required')
            self.activity.stop_notice.banner.setText(message);self.activity.stop_notice.banner.show()
        self.info_toggle.setChecked(incomplete)

    def toggle_activity(self,checked):
        self.info_toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.summary.setVisible(bool(checked and self.result))
        self.activity.log.setVisible(bool(checked and self.activity.started is not None))
        self.log_splitter.setVisible(bool(checked))

    def show_result(self,result,keep_session=False):
        if not keep_session:
            self.session=make_session([result]);self.configure_navigation()
        self.result=result;self.highlight=None;self.selected=selected_ids(result)
        result['selected_ids']=set(self.selected)
        self.details.clear()
        snapshot=result['snapshot'];m=snapshot.metadata
        first='first analyzed appearance' if has_skipped_frames(self.session) else 'first appearance'
        scope=(f'Movie-global IDs • Statistics: {first} only' if self.session['id_scope']=='movie-global' else 'Frame-local IDs')
        approach=DETECTION_APPROACHES.get(result.get('audit',{}).get('detection_start_mode'),'')
        approach_text=(' • '+approach) if approach else ''
        self.result_label.setText(f'Result: {Path(m.get("source_path","input")).name} • Frame {m["frame_0based"]+1} • Channel 1 • {result.get("origin","ai")} • {scope}{approach_text}')
        revisions=result.get('audit',{}).get('detection_linking',{}).get('revisions',[])
        revision_text=''.join(f'\nLocal detection repair (NumPy y/x bounds {r["bbox_yx"]}): {r["reason"]}' for r in revisions)
        self.summary.setPlainText(result['algorithm']+'\n'+result['summary']+'\n'+'\n'.join(result['warnings'])+revision_text)
        self.summary.setVisible(self.info_toggle.isChecked())
        for box,flag in ((self.include_uncertain,'uncertain'),(self.include_edge,'edge')):
            observations=first_observations(self.session).values() if self.session['id_scope']=='movie-global' else [(result,p) for p in result['particles']]
            flagged=[(r,p) for r,p in observations if p[flag]]
            with QtCore.QSignalBlocker(box):box.setChecked(bool(flagged and all(p['id'] in selected_ids(r) for r,p in flagged)))
        with QtCore.QSignalBlocker(self.table):
            self.table.clearSelection();self.table.setRowCount(len(result['particles']))
            for row,p in enumerate(result['particles']):
                use=QtWidgets.QTableWidgetItem();use.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsUserCheckable)
                use.setData(QtCore.Qt.UserRole,p['id']);use.setCheckState(QtCore.Qt.Checked if p['id'] in self.selected else QtCore.Qt.Unchecked);self.table.setItem(row,0,use)
                for col,(key,_) in enumerate(COLUMNS,1):
                    value=('Edge + uncertain' if p['edge'] and p['uncertain'] else 'Edge' if p['edge'] else 'Uncertain' if p['uncertain'] else 'Stable') if key=='status' else p.get(key)
                    if key=='detection' and 'detection_changed' in p:
                        local=', '.join('#'+str(i) for i in p.get('source_local_ids',[]))
                        value=('Revised '+local if local else 'Added') if p['detection_changed'] else 'Reused '+local
                    text='—' if value is None else str(value) if isinstance(value,(str,int)) else f'{value:.4g}'
                    item=QtWidgets.QTableWidgetItem(text)
                    if key in ('id','status'):
                        color='#185cc8' if p['edge'] else '#9a238a' if p['uncertain'] else '#087d4b'
                        item.setForeground(QtGui.QColor(color))
                    item.setToolTip(p['reason']+'\nRelative height/volume need at least 8 nearby background pixels. Edge geometry is incomplete.')
                    self.table.setItem(row,col,item)
            self.table.resizeColumnsToContents()
        self.update_preview();self.refresh_controls()

    def bulk_selection(self,*_):
        if not self.result:return
        if self.session['id_scope']=='movie-global':
            chosen={ident for ident,(_,p) in first_observations(self.session).items() if
                (not p['uncertain'] or self.include_uncertain.isChecked()) and (not p['edge'] or self.include_edge.isChecked())}
            for r in self.session['frames']:r['selected_ids']=chosen & {p['id'] for p in r['particles']}
            self.show_result(self.result,keep_session=True);return
        self.selected={p['id'] for p in self.result['particles'] if
            (not p['uncertain'] or self.include_uncertain.isChecked()) and (not p['edge'] or self.include_edge.isChecked())}
        set_session_selection(self.session,self.result,self.selected)
        with QtCore.QSignalBlocker(self.table):
            for row in range(self.table.rowCount()):
                item=self.table.item(row,0);item.setCheckState(QtCore.Qt.Checked if item.data(QtCore.Qt.UserRole) in self.selected else QtCore.Qt.Unchecked)
        self.update_preview();self.refresh_controls()

    def selection_changed(self,item):
        if item.column()!=0 or not self.result:return
        ident=item.data(QtCore.Qt.UserRole)
        if item.checkState()==QtCore.Qt.Checked:self.selected.add(ident)
        else:self.selected.discard(ident)
        set_session_selection(self.session,self.result,self.selected)
        self.update_preview();self.refresh_controls()

    def pick(self,ident):
        for row in range(self.table.rowCount()):
            if self.table.item(row,0).data(QtCore.Qt.UserRole)==ident:
                self.table.selectRow(row);self.table.scrollToItem(self.table.item(row,0));break

    def row_selected(self):
        rows=self.table.selectionModel().selectedRows()
        if not self.result or not rows:return
        ident=self.table.item(rows[0].row(),0).data(QtCore.Qt.UserRole);self.highlight=ident
        p=next(p for p in self.result['particles'] if p['id']==ident)
        baseline='Local background unavailable; relative height and volume are not calculated.' if p['background_nm'] is None else f'Local background from {p["background_pixels"]} pixels.'
        self.details.setText(f'Particle {ident}: {p["reason"]} {baseline}'+(' Edge / partial: size is incomplete.' if p['edge'] else ''))
        self.update_preview()

    def update_preview(self,*_):
        if not self.result:return
        rows=self.result['particles']; snapshot=self.result['snapshot']
        records=() if self.original.isChecked() else rows
        selected,show_ids,highlight=set(self.selected),self.show_ids.isChecked(),self.highlight
        image=render(snapshot,records,selected,show_ids,highlight)
        labels=None if self.original.isChecked() else self.result['labels']
        self.preview.set_image(image,labels)
        def render_large():
            from ai_image_review import pil_pixmap
            return [('Particles / IDs',pil_pixmap(render(snapshot,records,selected,show_ids,highlight,
                                                         long_edge=max(snapshot.image.shape))))]
        self.image_review.set_images([('Particles / IDs',self.preview.pixmap)],self.result_label.text(),
                                     render=render_large if max(snapshot.image.shape)>1000 else None)
        uncertain=sum(p['uncertain'] for p in rows);edge=sum(p['edge'] for p in rows)
        text=f'{len(rows)} candidates in this frame • {len(self.selected)} selected • {uncertain} uncertain • {edge} edge / partial. ×: unchecked.'
        if self.result.get('origin')=='skipped':
            info=self.result['audit']['frame_sampling']
            text=(f'Skipped — {info["overlap_fraction"]:.1%} overlap with representative frame {info["reference_frame_0based"]+1}. '
                  'Image only: no particle detection, IDs or measurements for this frame.')
        changes=self.result.get('audit',{}).get('detection_linking')
        if changes:
            text+=f' Normal detection: {changes["reused_particles"]} contours reused; {changes["revised_particles"]} revised; {len(changes["revisions"])} local repair region(s).'
        overlap=self.result.get('audit',{}).get('normal_detection',{}).get('parameters',{}).get('overlap_reuse',{})
        if overlap.get('mode') in ('reused','local'):
            text+=f' Field overlap: {overlap["overlap_fraction"]:.1%} with frame {overlap["reference_frame_0based"]+1}; '
            text+=f'{overlap["reused_particles"]} reference contours reused, {overlap["new_local_particles"]} locally detected.'
        if self.session['id_scope']=='movie-global':
            skipped=sum(r.get('origin')=='skipped' for r in self.session['frames'])
            first='first analyzed appearance' if skipped else 'first appearance'
            text+=f' Movie: {len(first_observations(self.session))} unique IDs; {len(statistics_rows(self.session))} included in statistics ({first} only).'
            if skipped:text+=f' {len(self.session["frames"])-skipped} analyzed / {skipped} skipped frames.'
        self.counts.setText(text)
        self.counts.setToolTip('Z uses the captured height datum; Δbg uses nearby background. Volume is a signed integral. Unchecked particles remain visible.')

    def pick_large_image(self,index,x,y):
        if not self.result or self.original.isChecked():return
        labels=self.result['labels'];h,w=labels.shape
        ident=int(labels[h-1-min(h-1,max(0,int(y*h))),min(w-1,max(0,int(x*w)))])
        if ident:self.pick(ident)

    def apply_to_main(self):
        if not self.apply_button.isEnabled():return
        warnings=apply_warnings(self.session)
        if warnings and not self.confirm_apply(warnings):
            self.status.setText('Apply cancelled. Review the flagged particles, adjust Use, or Refine, then Apply again.');return
        try:
            self.owner.applyAnalysisSession(self.session)
        except Exception as exc:
            self.status.setText('Cannot apply: '+str(exc));return
        self.status.setText('Applied to Particle Analysis. Further review changes require Apply again.')
        self.owner.show();self.owner.raise_();self.owner.activateWindow()

    def confirm_apply(self,warnings):
        """Human review is required before Apply; ask explicitly when flags remain."""
        text=('This AI result still needs human review:\n\n• '+'\n• '.join(warnings)+
              '\n\nApply it to Particle Analysis anyway? Uncheck Use for uncertain particles to leave them out.')
        answer=QtWidgets.QMessageBox.question(self,'Apply AI result?',text,
            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
        return answer==QtWidgets.QMessageBox.Yes

    def export(self,suffix,session=None):
        session=self.session if session is None else session
        if not session or self.worker is not None or self.preparing:return
        meta=session['frames'][0]['snapshot'].metadata
        source=Path(meta.get('source_path') or 'particles.asd')
        scope=f'_F{meta["frame_0based"]+1:04d}' if len(session['frames'])==1 else '_all_frames'
        target=source.with_name(source.stem+scope+'_particles'+suffix)
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Export Particle Analysis',str(target),'CSV (*.csv)' if suffix=='.csv' else 'JSON (*.json)')
        if not path:return
        if Path(path).suffix.lower()!=suffix:path+=suffix
        self.stop.clear();self.activity.start('Saving analyzed frames and particle choices…')
        def job(progress):
            export_session(session,path);return path
        self.launch_worker(job,lambda saved:self.activity.finish('Saved: '+saved),'export')

    def cancel(self):
        if self.worker_mode!='export' and (self.worker is not None or self.preparing):
            self.stop.set();self.activity.cancel()

    def finished(self):
        self.worker.deleteLater();self.worker=None;self.worker_mode=None;self.refresh_controls()
        if self.closing:self.close()

    def reject(self):self.close()

    def closeEvent(self,event):
        if self.worker is not None or self.preparing:
            self.closing=True;self.cancel();event.ignore();return
        self.activity.timer.stop();self.timer.stop();event.accept()

    def showEvent(self,event):
        self.timer.start(500);self.refresh_context();super().showEvent(event)
