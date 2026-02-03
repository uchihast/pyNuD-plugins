import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDesktopWidget, QPushButton, QFileDialog, QComboBox, QSpinBox, QSlider, QProgressBar, QMessageBox, QMenuBar, QAction, QTextEdit, QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem, QSplitter, QToolBar, QScrollArea
from PyQt5.QtCore import Qt, QDateTime, QSettings
import os
import vtk
import numpy as np
from PyQt5.QtCore import QTimer

# VTK 9.x compatibility
try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError:
        from vtkmodules.vtkRenderingQt import QVTKRenderWindowInteractor

PLUGIN_NAME = "Normal Mode Analysis"


class NMA_Window(QMainWindow):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("NMA(Normal Mode Analysis) Simulation")
        
        # Windows固有の設定
        #import sys
        #if sys.platform.startswith('win'):
            # Windowsでの安定性向上のための設定
        #    self.setAttribute(Qt.WA_OpaquePaintEvent, True)
       #     self.setAttribute(Qt.WA_NoSystemBackground, True)
        
        # スタンドアロンアプリケーションなのでwindow_managerは使用しない
        
        # ウィンドウの位置とサイズを復元
        self.settings = QSettings("pyNuD", "NMA_Window")
        self.restore_geometry()
        
        # 設定が保存されていない場合はデフォルトサイズを使用
        if not self.settings.contains("geometry"):
            # 画面サイズの60%で表示 → 1.2倍に拡大
            desktop = QDesktopWidget()
            screen_geometry = desktop.screenGeometry()
            width = int(screen_geometry.width() * 0.6 * 1.2)
            height = int(screen_geometry.height() * 0.7 * 1.2)  # 縦を70%に拡大
            self.setMinimumSize(600, 600)  # 最小サイズも縦に拡大
            self.resize(width, height)
            self.center_on_screen()
        
        # ファイルダイアログの初期ディレクトリ管理
        self.last_import_dir = ""
        
        # メニューバー作成フラグ
        self.menu_bar_created = False
        
        # メインウィジェットとレイアウト
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # メニューバーを作成（setCentralWidgetの後）
        self.create_menu_bar()
        
        # スクロール可能な左パネル（ボタン）
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setMinimumWidth(300)
        left_scroll_area.setMaximumWidth(400)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # Import PDB File...ボタン（左寄せ）
        self.import_btn = QPushButton("Import PDB File...")
        self.import_btn.setMinimumHeight(35)
        self.import_btn.setFixedWidth(200)
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.import_btn.clicked.connect(self.import_pdb_file)
        left_layout.addWidget(self.import_btn, alignment=Qt.AlignLeft)

        # PDBファイル名表示ラベル
        self.pdb_label = QLabel("PDB Name: (none)")
        self.pdb_label.setStyleSheet("color: #666; font-size: 12px;")
        left_layout.addWidget(self.pdb_label, alignment=Qt.AlignLeft)

        # Style
        self.style_label = QLabel("Style:")
        left_layout.addWidget(self.style_label, alignment=Qt.AlignLeft)
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "Ball & Stick", "Stick Only", "Spheres", "Points"
        ])
        self.style_combo.currentTextChanged.connect(self.update_pdb_display)
        self.style_combo.setFixedWidth(200)
        left_layout.addWidget(self.style_combo, alignment=Qt.AlignLeft)

        # Color
        self.color_label = QLabel("Color:")
        left_layout.addWidget(self.color_label, alignment=Qt.AlignLeft)
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "By Element", "By Chain", "Single Color", "By B-Factor"
        ])
        self.color_combo.currentTextChanged.connect(self.update_pdb_display)
        self.color_combo.setFixedWidth(200)
        left_layout.addWidget(self.color_combo, alignment=Qt.AlignLeft)

        # Atom選択
        self.atom_label = QLabel("Show:")
        left_layout.addWidget(self.atom_label, alignment=Qt.AlignLeft)
        self.atom_combo = QComboBox()
        self.atom_combo.addItems([
            "All Atoms", "Heavy Atoms", "Backbone", "C", "N", "O"
        ])
        self.atom_combo.currentTextChanged.connect(self.update_pdb_display)
        self.atom_combo.setFixedWidth(200)
        left_layout.addWidget(self.atom_combo, alignment=Qt.AlignLeft)

                # NMA Parameters
        self.nma_label = QLabel("NMA Parameters:")
        self.nma_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(self.nma_label, alignment=Qt.AlignLeft)
        
        # Atom Selection for NMA
        self.nma_atom_label = QLabel("Atom Selection:")
        left_layout.addWidget(self.nma_atom_label, alignment=Qt.AlignLeft)
        self.nma_atom_combo = QComboBox()
        self.nma_atom_combo.addItems([
            "calpha", "all"
        ])
        self.nma_atom_combo.setCurrentText("calpha")
        self.nma_atom_combo.setFixedWidth(200)
        left_layout.addWidget(self.nma_atom_combo, alignment=Qt.AlignLeft)

        # Cutoff Distance
        self.cutoff_label = QLabel("Cutoff Distance (Å):")
        left_layout.addWidget(self.cutoff_label, alignment=Qt.AlignLeft)
        self.cutoff_spinbox = QSpinBox()
        self.cutoff_spinbox.setRange(5, 30)
        self.cutoff_spinbox.setValue(15)
        self.cutoff_spinbox.setSuffix(" Å")
        self.cutoff_spinbox.setFixedWidth(200)
        left_layout.addWidget(self.cutoff_spinbox, alignment=Qt.AlignLeft)

        # Number of Modes
        self.modes_label = QLabel("Number of Modes:")
        left_layout.addWidget(self.modes_label, alignment=Qt.AlignLeft)
        self.modes_spinbox = QSpinBox()
        self.modes_spinbox.setRange(1, 100)
        self.modes_spinbox.setValue(20)
        self.modes_spinbox.setFixedWidth(200)
        left_layout.addWidget(self.modes_spinbox, alignment=Qt.AlignLeft)

        # Run NMA Simulation Button
        self.run_nma_btn = QPushButton("Run NMA Simulation")
        self.run_nma_btn.setMinimumHeight(35)
        self.run_nma_btn.setFixedWidth(200)  # Import PDB Fileボタンと同じサイズ
        self.run_nma_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_nma_btn.clicked.connect(self.run_nma_simulation)
        self.run_nma_btn.setEnabled(False)
        left_layout.addWidget(self.run_nma_btn, alignment=Qt.AlignLeft)

        # Progress Bar for NMA calculation
        self.nma_progress = QProgressBar()
        self.nma_progress.setMinimum(0)
        self.nma_progress.setMaximum(100)
        self.nma_progress.setValue(0)
        self.nma_progress.setVisible(True)
        left_layout.addWidget(self.nma_progress, alignment=Qt.AlignLeft)

        # NMA Results Display
        self.results_label = QLabel("NMA Results:")
        self.results_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(self.results_label, alignment=Qt.AlignLeft)

        # Mode Selection
        self.mode_label = QLabel("Display Mode:")
        left_layout.addWidget(self.mode_label, alignment=Qt.AlignLeft)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["None"])
        self.mode_combo.currentTextChanged.connect(self.display_nma_mode)
        self.mode_combo.setFixedWidth(200)  # Showと同じサイズ
        left_layout.addWidget(self.mode_combo, alignment=Qt.AlignLeft)

        # Amplitude Slider
        self.amplitude_label = QLabel("Amplitude:")
        left_layout.addWidget(self.amplitude_label, alignment=Qt.AlignLeft)
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setRange(-100, 100)
        self.amplitude_slider.setValue(0)
        self.amplitude_slider.valueChanged.connect(self.update_nma_display)
        left_layout.addWidget(self.amplitude_slider, alignment=Qt.AlignLeft)
        
        # Save Deformed Structure Button
        self.save_deformed_btn = QPushButton("Save Deformed Structure")
        self.save_deformed_btn.setMinimumHeight(35)
        self.save_deformed_btn.setFixedWidth(200)
        self.save_deformed_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.save_deformed_btn.clicked.connect(self.save_deformed_structure)
        self.save_deformed_btn.setEnabled(False)
        left_layout.addWidget(self.save_deformed_btn, alignment=Qt.AlignLeft)
        


        # Store NMA results
        self.nma_results = None
        self.nma_selection = None
        self.is_calculating = False
        
        # 静的なPDBと、変形後のNMAモデルのアクターを別々に管理する
        self.pdb_actors = []
        self.nma_actors = []

        left_layout.addStretch()
        
        # スクロールエリアに左パネルを設定
        left_scroll_area.setWidget(left_panel)
        main_layout.addWidget(left_scroll_area)

        # 右パネル（VTK 3Dビュー）
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.Initialize()
        
        # NMAビュー（右側）
        self.nma_vtk_widget = QVTKRenderWindowInteractor(central_widget)
        self.nma_renderer = vtk.vtkRenderer()
        self.nma_vtk_widget.GetRenderWindow().AddRenderer(self.nma_renderer)
        self.nma_vtk_widget.Initialize()
        
        # カスタムInteractorStyleを作成して適用
        # Original structure用（メイン）
        self.main_style = SynchronizedInteractorStyle(self.renderer, self.nma_renderer)
        self.vtk_widget.SetInteractorStyle(self.main_style)
        
        # Deformed structure用（セカンダリ）
        self.nma_style = SynchronizedInteractorStyle(self.nma_renderer, self.renderer)
        self.nma_vtk_widget.SetInteractorStyle(self.nma_style)
        
        # 初期カメラ位置を同期（少し遅延させて確実に同期）
        QTimer.singleShot(100, self.sync_initial_cameras)
        
        # ビューアーを水平レイアウトで配置
        viewer_layout = QHBoxLayout()
        
        # PDBビュー（左側）
        pdb_view_layout = QVBoxLayout()
        pdb_label = QLabel("Original Structure")
        pdb_label.setAlignment(Qt.AlignCenter)
        pdb_label.setStyleSheet("color: white; font-weight: bold; background-color: #2196F3; padding: 5px;")
        pdb_view_layout.addWidget(pdb_label)
        
        # PDBビューをスクロール可能にする
        pdb_scroll_area = QScrollArea()
        pdb_scroll_area.setWidgetResizable(True)
        pdb_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pdb_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pdb_scroll_area.setWidget(self.vtk_widget)
        pdb_view_layout.addWidget(pdb_scroll_area)
        
        # NMAビュー（右側）
        nma_view_layout = QVBoxLayout()
        nma_label = QLabel("NMA Deformed Structure")
        nma_label.setAlignment(Qt.AlignCenter)
        nma_label.setStyleSheet("color: white; font-weight: bold; background-color: #FF9800; padding: 5px;")
        nma_view_layout.addWidget(nma_label)
        
        # NMAビューをスクロール可能にする
        nma_scroll_area = QScrollArea()
        nma_scroll_area.setWidgetResizable(True)
        nma_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        nma_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        nma_scroll_area.setWidget(self.nma_vtk_widget)
        nma_view_layout.addWidget(nma_scroll_area)
        
        viewer_layout.addLayout(pdb_view_layout)
        viewer_layout.addLayout(nma_view_layout)
        
        main_layout.addLayout(viewer_layout)
        
        # ウィンドウの初期化完了後にメニューバーを再設定
        QTimer.singleShot(100, self.ensure_menu_bar)
        
        # macOSでのメニューバー表示を確実にする
        if sys.platform == "darwin":
            QTimer.singleShot(200, self.force_menu_bar_visible)
        
        # ウィンドウが表示された後にメニューバーを再設定
        QTimer.singleShot(500, self.final_menu_bar_check)

    def showEvent(self, event):
        """ウィンドウが表示される時にメニューバーを確実に設定"""
        super().showEvent(event)
        # ウィンドウが表示された後にメニューバーを再設定
        QTimer.singleShot(100, self.ensure_menu_bar_after_show)

    def ensure_menu_bar_after_show(self):
        """ウィンドウ表示後のメニューバー確保"""
        try:
            menubar = self.menuBar()
            if not menubar.isVisible() or len(menubar.actions()) == 0:
                self.menu_bar_created = False  # フラグをリセット
                self.create_menu_bar()
        except Exception as e:
            pass  # エラーは無視

    def ensure_menu_bar(self):
        """メニューバーが確実に表示されるようにする"""
        try:
            menubar = self.menuBar()
            if not menubar.isVisible():
                menubar.setVisible(True)
                self.menu_bar_created = False  # フラグをリセット
                self.create_menu_bar()
        except Exception as e:
            pass  # エラーは無視

    def force_menu_bar_visible(self):
        """macOSでメニューバーを強制的に表示"""
        try:
            menubar = self.menuBar()
            menubar.setVisible(True)
            menubar.setNativeMenuBar(False)  # ネイティブメニューバーを無効化
        except Exception as e:
            pass  # エラーは無視

    def final_menu_bar_check(self):
        """最終的なメニューバーの確認と設定"""
        try:
            menubar = self.menuBar()
            if not menubar.isVisible() or menubar.actions() == []:
                self.menu_bar_created = False  # フラグをリセット
                self.create_menu_bar()
                menubar.setVisible(True)
        except Exception as e:
            pass  # エラーは無視

    def create_menu_bar(self):
        """メニューバーを作成"""
        # 既に作成済みの場合はスキップ
        if self.menu_bar_created:
            return
            
        try:
            menubar = self.menuBar()
            
            # 既にメニューが存在する場合はクリア
            menubar.clear()
            
            # メニューバーを明示的に表示
            menubar.setVisible(True)
            
            # Helpメニュー
            help_menu = menubar.addMenu("&Help")
            
            # NMA Help
            nma_help_action = QAction("&NMA Help", self)
            nma_help_action.setShortcut("F1")
            nma_help_action.triggered.connect(self.show_nma_help)
            help_menu.addAction(nma_help_action)
            
            help_menu.addSeparator()
            
            # About
            about_action = QAction("&About NMA", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
            
            # メニューバーを強制的に更新
            menubar.update()
            
            # 作成完了フラグを設定
            self.menu_bar_created = True
            
        except Exception as e:
            # フォールバック: ツールバーにヘルプボタンを追加
            self.create_fallback_help_button()

    def create_fallback_help_button(self):
        """フォールバック用のヘルプボタンを作成"""
        try:
            # ツールバーを作成
            toolbar = self.addToolBar("Help")
            
            # ヘルプボタンを追加
            help_action = QAction("Help (F1)", self)
            help_action.setShortcut("F1")
            help_action.triggered.connect(self.show_nma_help)
            toolbar.addAction(help_action)
            
            # Aboutボタンを追加
            about_action = QAction("About NMA", self)
            about_action.triggered.connect(self.show_about)
            toolbar.addAction(about_action)
            
            pass  # フォールバックツールバー作成完了
            
        except Exception as e:
            pass  # エラーは無視

    def show_nma_help(self):
        """NMAヘルプウィンドウを表示"""
        if not hasattr(self, 'help_window') or self.help_window is None:
            self.help_window = NMAHelpWindow(self)
        
        # 独立したウィンドウとして表示
        self.help_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.help_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()


    def show_about(self):
        """Aboutダイアログを表示"""
        about_text = """
        <b>pyNuD NMA (Normal Mode Analysis)</b><br><br>
        <b>Version:</b> 1.0.0<br>
        <b>Description:</b> Normal Mode Analysis tool for protein dynamics<br>
        <b>Features:</b><br>
        • PDB file import and visualization<br>
        • Normal Mode Analysis calculation using ProDy<br>
        • Interactive mode visualization<br>
        • Deformed structure export<br>
        • Synchronized dual-view display<br><br>
        <b>Dependencies:</b><br>
        • ProDy: Normal Mode Analysis library<br>
        • VTK: 3D visualization<br>
        • PyQt5: User interface<br>
        • NumPy: Numerical computations
        """
        
        QMessageBox.about(self, "About pyNuD NMA", about_text)

    def center_on_screen(self):
        frame_geom = self.frameGeometry()
        screen = QDesktopWidget().availableGeometry().center()
        frame_geom.moveCenter(screen)
        self.move(frame_geom.topLeft())
    
    def restore_geometry(self):
        """ウィンドウの位置とサイズを復元"""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except Exception:
            pass  # 復元に失敗した場合は無視
    
    def save_geometry(self):
        """ウィンドウの位置とサイズを保存"""
        try:
            geometry = self.saveGeometry()
            self.settings.setValue("geometry", geometry)
        except Exception:
            pass  # 保存に失敗した場合は無視
    
    def closeEvent(self, event):
        """ウィンドウが閉じられる時の処理"""
        # 設定を保存
        self.save_geometry()
        
        # ツールバーアクションのハイライトを解除（プラグインとして開かれた場合／pyNuDから開かれた場合）
        try:
            if self.main_window is not None and hasattr(self.main_window, 'plugin_actions'):
                action = self.main_window.plugin_actions.get(PLUGIN_NAME)
                if action is not None and hasattr(self.main_window, 'setActionHighlight'):
                    self.main_window.setActionHighlight(action, False)
            else:
                import globalvals as gv
                if hasattr(gv, 'main_window') and gv.main_window:
                    if hasattr(gv.main_window, 'setActionHighlight') and hasattr(gv.main_window, 'nma_action'):
                        gv.main_window.setActionHighlight(gv.main_window.nma_action, False)
        except Exception:
            pass  # スタンドアロン起動の場合は無視
        
        # 親クラスのcloseEventを呼び出し
        super().closeEvent(event)

    def import_pdb_file(self):
        initial_dir = self.last_import_dir if self.last_import_dir else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDB File", initial_dir, "PDB files (*.pdb);;All files (*)",
            options=QFileDialog.DontUseNativeDialog)
        if file_path:
            self.last_import_dir = os.path.dirname(file_path)
            self.current_pdb_path = file_path
            
            # NMA計算結果を初期化
            self.nma_results = None
            self.nma_selection = None
            self.is_calculating = False
            
            # NMA表示をクリア
            for actor in self.nma_actors:
                self.nma_renderer.RemoveActor(actor)
            self.nma_actors.clear()
            self.nma_renderer.RemoveAllViewProps()
            self.nma_vtk_widget.GetRenderWindow().Render()
            
            # モード選択UIをリセット
            self.mode_combo.clear()
            self.mode_combo.addItem("None")
            
            # プログレスバーをリセット
            self.nma_progress.setValue(0)
            
            # ボタン状態をリセット
            self.run_nma_btn.setText("Run NMA Simulation")
            self.run_nma_btn.setEnabled(True)
            self.save_deformed_btn.setEnabled(False)
            
            # 新しいPDBを表示
            self.display_pdb(file_path)
            self.pdb_label.setText(f"PDB Name: {os.path.basename(file_path)}")

    def update_pdb_display(self):
        if hasattr(self, 'current_pdb_path') and self.current_pdb_path:
            self.display_pdb(self.current_pdb_path)
            # NMAモードが選択されている場合、その表示も更新する
            self.display_nma_mode()

    def parse_pdb_atoms(self, file_path):
        atoms = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        atom = {
                            'serial': int(line[6:11]),
                            'name': line[12:16].strip(),
                            'residue_name': line[17:20].strip(),
                            'chain_id': line[21:22].strip(),
                            'residue_id': int(line[22:26]),
                            'x': float(line[30:38]) / 10.0,
                            'y': float(line[38:46]) / 10.0,
                            'z': float(line[46:54]) / 10.0,
                            'element': line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0]
                        }
                        atoms.append(atom)
                    except Exception:
                        continue
        return atoms

    def _filter_atoms_by_selection(self, atoms, selection):
        """Showの選択肢に基づいて原子をフィルタリング"""
        if selection == "All Atoms":
            return atoms
        elif selection == "Heavy Atoms":
            # H以外の原子を選択
            return [atom for atom in atoms if atom['element'] != 'H']
        elif selection == "Backbone":
            # N, CA, C原子のみを選択
            return [atom for atom in atoms if atom['name'] in ['N', 'CA', 'C']]
        elif selection == "C":
            # 炭素原子のみを選択
            return [atom for atom in atoms if atom['element'] == 'C']
        elif selection == "N":
            # 窒素原子のみを選択
            return [atom for atom in atoms if atom['element'] == 'N']
        elif selection == "O":
            # 酸素原子のみを選択
            return [atom for atom in atoms if atom['element'] == 'O']
        else:
            return atoms

    def _create_actors_from_atom_list(self, atoms, force_points=False, skip_bonds=False):
        import numpy as np
        """
        原子情報のリストを受け取り、現在のUI設定に基づいてVTKアクターのリストを生成するヘルパー関数。
        force_points: Trueなら必ず点表示
        skip_bonds: Trueならボンド描画を省略
        """
        if not atoms:
            return []

        # UIから現在の設定を取得
        style = self.style_combo.currentText()
        color_mode = self.color_combo.currentText()
        if force_points:
            style = "Points"
        
        actors = []
        atom_indices = {atom['serial']: idx for idx, atom in enumerate(atoms)}

        # 結合情報を事前にパース（CONECTレコードがある場合）
        bonds = []
        if hasattr(self, 'parsed_bonds'):
             bonds = self.parsed_bonds

        def get_color(atom):
            # PDB表示の場合は通常の色設定
            if color_mode == "By Element":
                table = {'C': (0.3, 0.3, 0.3), 'O': (1.0, 0.3, 0.3), 'N': (0.3, 0.3, 1.0), 'H': (0.9, 0.9, 0.9), 'S': (1.0, 1.0, 0.3), 'P': (1.0, 0.5, 0.0)}
                return table.get(atom['element'], (0.7, 0.7, 0.7))
            elif color_mode == "By Chain":
                # チェーンIDから色を生成（単純なハッシュ）
                chain_hash = ord(atom['chain_id'][0]) if atom['chain_id'] else 0
                # HSVからRGBに変換（簡易版）
                hue = (chain_hash * 30) % 360
                h = hue / 60.0
                i = int(h)
                f = h - i
                p = 0.0
                q = 0.8 * (1.0 - f)
                t = 0.8 * f
                
                if i == 0:
                    r, g, b = 0.8, t, p
                elif i == 1:
                    r, g, b = q, 0.8, p
                elif i == 2:
                    r, g, b = p, 0.8, t
                elif i == 3:
                    r, g, b = p, q, 0.8
                elif i == 4:
                    r, g, b = t, p, 0.8
                else:
                    r, g, b = 0.8, p, q
                
                return (r, g, b)
            elif color_mode == "Single Color":
                return (0.5, 0.7, 0.9)
            elif color_mode == "By B-Factor":
                # B-Factorの値を色に反映（仮の実装）
                bfactor = atom.get('bfactor', 20.0) if isinstance(atom, dict) else 20.0
                # B-Factorを0-1の範囲に正規化（20-80の範囲を想定）
                normalized_bfactor = max(0.0, min(1.0, (bfactor - 20.0) / 60.0))
                return (normalized_bfactor, 0.5, 1.0 - normalized_bfactor)
            else: # デフォルト
                return (0.7, 0.7, 0.7)

        # --- スタイルに応じたアクター生成 ---
        if style in ["Spheres", "Ball & Stick"]:
            radius = 0.1 if style == "Spheres" else 0.08
            for atom in atoms:
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(atom['x'], atom['y'], atom['z'])
                sphere.SetRadius(radius)
                sphere.SetThetaResolution(16)
                sphere.SetPhiResolution(16)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*get_color(atom))
                actors.append(actor)

        if style in ["Stick Only", "Ball & Stick"] and not skip_bonds:
            # 結合の描画ロジック
            if bonds:
                for src, dst in bonds:
                    if src in atom_indices and dst in atom_indices:
                        a1 = atoms[atom_indices[src]]
                        a2 = atoms[atom_indices[dst]]
                        line = vtk.vtkLineSource()
                        line.SetPoint1(a1['x'], a1['y'], a1['z'])
                        line.SetPoint2(a2['x'], a2['y'], a2['z'])
                        tube = vtk.vtkTubeFilter()
                        tube.SetInputConnection(line.GetOutputPort())
                        tube.SetRadius(0.03)
                        tube.SetNumberOfSides(8)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(tube.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
                        actors.append(actor)
            else:
                backbone_atoms = [a for a in atoms if a['name'] in ("N", "CA", "C")]
                backbone_atoms.sort(key=lambda a: (a['chain_id'], a['residue_id']))
                for i in range(len(backbone_atoms) - 1):
                    a1 = backbone_atoms[i]
                    a2 = backbone_atoms[i+1]
                    if a1['chain_id'] == a2['chain_id'] and a2['residue_id'] == a1['residue_id'] + 1:
                        d = np.linalg.norm(np.array([a1['x'], a1['y'], a1['z']]) - np.array([a2['x'], a2['y'], a2['z']]))
                        if d < 0.4:
                            line = vtk.vtkLineSource()
                            line.SetPoint1(a1['x'], a1['y'], a1['z'])
                            line.SetPoint2(a2['x'], a2['y'], a2['z'])
                            tube = vtk.vtkTubeFilter()
                            tube.SetInputConnection(line.GetOutputPort())
                            tube.SetRadius(0.03)
                            tube.SetNumberOfSides(8)
                            mapper = vtk.vtkPolyDataMapper()
                            mapper.SetInputConnection(tube.GetOutputPort())
                            actor = vtk.vtkActor()
                            actor.SetMapper(mapper)
                            actor.GetProperty().SetColor(0.7, 0.7, 0.7)
                            actors.append(actor)

        if style == "Wireframe":
            # Wireframe表示
            for atom in atoms:
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(atom['x'], atom['y'], atom['z'])
                sphere.SetRadius(0.1)
                sphere.SetThetaResolution(8)
                sphere.SetPhiResolution(8)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetColor(*get_color(atom))
                actors.append(actor)

        if style == "Simple Cartoon":
            # Simple Cartoon表示（backboneのみを太い線で表示）
            backbone_atoms = [a for a in atoms if a['name'] in ("N", "CA", "C")]
            backbone_atoms.sort(key=lambda a: (a['chain_id'], a['residue_id']))
            for i in range(len(backbone_atoms) - 1):
                a1 = backbone_atoms[i]
                a2 = backbone_atoms[i+1]
                if a1['chain_id'] == a2['chain_id'] and a2['residue_id'] == a1['residue_id'] + 1:
                    d = np.linalg.norm(np.array([a1['x'], a1['y'], a1['z']]) - np.array([a2['x'], a2['y'], a2['z']]))
                    if d < 0.4:
                        line = vtk.vtkLineSource()
                        line.SetPoint1(a1['x'], a1['y'], a1['z'])
                        line.SetPoint2(a2['x'], a2['y'], a2['z'])
                        tube = vtk.vtkTubeFilter()
                        tube.SetInputConnection(line.GetOutputPort())
                        tube.SetRadius(0.08)  # 太い線
                        tube.SetNumberOfSides(12)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(tube.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        # チェーンごとに色を変える
                        chain_hash = ord(a1['chain_id'][0]) if a1['chain_id'] else 0
                        hue = (chain_hash * 30) % 360
                        color = vtk.vtkColor3d()
                        color.SetHsv(hue, 0.8, 0.9)
                        actor.GetProperty().SetColor(*color.GetRgb())
                        actors.append(actor)

        if style == "Points":
            # VTKのvertex glyphで高速描画
            import vtk.util.numpy_support as ns
            import numpy as np
            points = vtk.vtkPoints()
            arr = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms], dtype=np.float32)
            for xyz in arr:
                points.InsertNextPoint(*xyz)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(poly)
            glyph.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 1)
            actor.GetProperty().SetPointSize(3)  # 小さく
            actor.GetProperty().SetRenderPointsAsSpheres(True)  # 丸い点
            actors.append(actor)
        return actors

    def display_pdb(self, file_path):
        # 既存のアクターをクリア
        for actor in self.pdb_actors:
            self.renderer.RemoveActor(actor)
        self.pdb_actors.clear()

        # PDBファイルをパース
        atoms = self.parse_pdb_atoms(file_path)
        
        # Showの選択肢に基づいて原子をフィルタリング
        show_selection = self.atom_combo.currentText()
        filtered_atoms = self._filter_atoms_by_selection(atoms, show_selection)
        
        # PDB表示ではNMA設定の影響を受けない（通常の表示）
        self.pdb_actors = self._create_actors_from_atom_list(filtered_atoms, force_points=False, skip_bonds=False)
        for actor in self.pdb_actors:
            self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def run_nma_simulation(self):
        """ProDyを使用してNMAシミュレーションを実行"""
        # 計算中フラグを設定
        self.is_calculating = True
        self.run_nma_btn.setText("Calculating...")
        self.run_nma_btn.setEnabled(False)
        self.nma_progress.setValue(0)
        
        try:
            # ProDyのインポート
            import prody
            
            # パラメーター取得
            atom_selection = self.nma_atom_combo.currentText()
            cutoff_distance = self.cutoff_spinbox.value()
            num_modes = self.modes_spinbox.value()
            
            print(f"NMA Simulation Parameters:")
            print(f"  Atom Selection: {atom_selection}")
            print(f"  Cutoff Distance: {cutoff_distance} Å")
            print(f"  Number of Modes: {num_modes}")
            
            # プログレス更新
            self.nma_progress.setValue(10)
            QApplication.processEvents()
            
            # PDBファイルをProDyで読み込み
            protein = prody.parsePDB(self.current_pdb_path)
            self.nma_progress.setValue(20)
            QApplication.processEvents()
            
            # 原子選択
            if atom_selection == "calpha":
                selection = protein.select('calpha')
            else:  # "all"
                selection = protein.select('protein')
                
            if selection is None:
                print("Error: No atoms selected for NMA calculation")
                self._reset_calculation_state()
                return
                
            print(f"Selected {len(selection)} atoms for NMA calculation")
            self.nma_progress.setValue(40)
            QApplication.processEvents()
            
            # NMAを初期化して計算を実行
            nma = prody.ANM(f'NMA for {os.path.basename(self.current_pdb_path)}')
            self.nma_progress.setValue(60)
            QApplication.processEvents()
            
            nma.buildHessian(selection, cutoff=cutoff_distance)
            self.nma_progress.setValue(80)
            QApplication.processEvents()
            
            nma.calcModes(n_modes=num_modes)
            self.nma_progress.setValue(90)
            QApplication.processEvents()
            
            # 結果の概要を表示
            print(f"\nNMA Results:")
            print(f"  Number of modes calculated: {len(nma)}")
            print(f"  Number of atoms: {len(selection)}")
            print(f"  Cutoff distance used: {cutoff_distance} Å")
            
            # 各モードの情報を表示
            for i, mode in enumerate(nma):
                print(f"  Mode {i+1}: Eigenvalue = {mode.getEigval():.6f}")
                
            # NMA結果を保存
            self.nma_results = nma
            self.nma_selection = selection
            
            # モード選択UIを更新
            self.mode_combo.clear()
            self.mode_combo.addItem("None")
            for i in range(len(nma)):
                eigenvalue = nma[i].getEigval()
                # 正のモードと負のモードを区別
                if eigenvalue > 0:
                    self.mode_combo.addItem(f"Mode +{i+1} (λ={eigenvalue:.4f})")
                else:
                    self.mode_combo.addItem(f"Mode -{i+1} (λ={eigenvalue:.4f})")
                
            print("\nNMA simulation completed successfully!")
            print("Select a mode from the dropdown to visualize the dynamics.")
            self.nma_progress.setValue(100)
            
            # 保存ボタンを有効化
            self.save_deformed_btn.setEnabled(True)
            
        except ImportError:
            print("Error: ProDy is not installed. Please install ProDy to run NMA simulations.")
            print("Installation: pip install prody")
            self.nma_progress.setValue(0)
        except Exception as e:
            print(f"Error during NMA simulation: {str(e)}")
            self.nma_progress.setValue(0)
        finally:
            self._reset_calculation_state()
    
    def _reset_calculation_state(self):
        """計算状態をリセット"""
        self.is_calculating = False
        self.run_nma_btn.setText("Run NMA Simulation")
        self.run_nma_btn.setEnabled(True)

    def display_nma_mode(self):
        """選択されたNMAモードで変形した構造をNMAビューに表示"""
        # 既存のNMAアクターをクリア
        for actor in self.nma_actors:
            self.nma_renderer.RemoveActor(actor)
        self.nma_actors.clear()

        selected_text = self.mode_combo.currentText()

        # "None"が選択されたら、NMAビューをクリア
        if selected_text == "None" or self.nma_results is None:
            self.nma_renderer.RemoveAllViewProps()
            self.nma_vtk_widget.GetRenderWindow().Render()
            return

        try:
            # モード番号を抽出（+1や-1から数値を取得）
            mode_text = selected_text.split()[1]
            mode_index = int(mode_text.replace('+', '').replace('-', '')) - 1
        except (IndexError, ValueError):
            return

        mode = self.nma_results[mode_index]
        amplitude = self.amplitude_slider.value()
        original_coords = self.nma_selection.getCoords()
        displacements = mode.getArray()
        if amplitude == 0:
            new_coords = original_coords
        else:
            num_atoms = len(self.nma_selection)
            displacements_3d = displacements.reshape(num_atoms, 3) * amplitude
            new_coords = original_coords + displacements_3d

        # atom selectionに応じてdeformed_atomsを作成
        atom_selection = self.nma_atom_combo.currentText() if hasattr(self, 'nma_atom_combo') else "calpha"
        deformed_atoms = []
        for i, atom in enumerate(self.nma_selection):
            if atom_selection == "calpha" and atom.getName() != "CA":
                continue
            deformed_atom = {
                'serial': atom.getSerial(),
                'name': atom.getName(),
                'residue_name': atom.getResname(),
                'chain_id': atom.getChid(),
                'residue_id': atom.getResnum(),
                'x': new_coords[i, 0] / 10.0,
                'y': new_coords[i, 1] / 10.0,
                'z': new_coords[i, 2] / 10.0,
                'element': atom.getElement() or atom.getName()[0]
            }
            deformed_atoms.append(deformed_atom)
        force_points = (atom_selection == "all")
        skip_bonds = (atom_selection == "all") or (atom_selection == "calpha")
        self.nma_actors = self._create_actors_from_atom_list(deformed_atoms, force_points=force_points, skip_bonds=skip_bonds)
        for actor in self.nma_actors:
            # calpha選択時は白色、all選択時は元の色設定を保持
            if atom_selection == "calpha":
                actor.GetProperty().SetColor(1, 1, 1)
            # all選択時は_create_actors_from_atom_list内で既に色設定されているため何もしない
            self.nma_renderer.AddActor(actor)
        self.nma_renderer.ResetCamera()
        self.nma_vtk_widget.GetRenderWindow().Render()

    def update_nma_display(self):
        """振幅スライダーが変更された時にNMA表示を更新"""
        if self.nma_results is not None and self.mode_combo.currentText() != "None":
            self.display_nma_mode()
    
    def sync_initial_cameras(self):
        """初期カメラ位置を同期"""
        if hasattr(self, 'renderer') and hasattr(self, 'nma_renderer'):
            # 両方のレンダラーが初期化された後に同期
            if hasattr(self, 'main_style'):
                self.main_style.sync_cameras(None, None)
            if hasattr(self, 'nma_style'):
                self.nma_style.sync_cameras(None, None)
    
    def save_deformed_structure(self):
        """変位した構造をPDBファイルとして保存"""
        if self.nma_results is None or self.mode_combo.currentText() == "None":
            QMessageBox.warning(self, "Warning", "No NMA results available for saving.")
            return
        
        try:
            # 現在のモードと振幅を取得
            selected_text = self.mode_combo.currentText()
            mode_text = selected_text.split()[1]
            mode_index = int(mode_text.replace('+', '').replace('-', '')) - 1
            mode = self.nma_results[mode_index]
            amplitude = self.amplitude_slider.value()
            
            # 変位した座標を計算
            original_coords = self.nma_selection.getCoords()
            displacements = mode.getArray()
            if amplitude == 0:
                new_coords = original_coords
            else:
                num_atoms = len(self.nma_selection)
                displacements_3d = displacements.reshape(num_atoms, 3) * amplitude
                new_coords = original_coords + displacements_3d
            
            # 保存先ファイル名を生成
            base_name = os.path.splitext(os.path.basename(self.current_pdb_path))[0]
            mode_num = mode_index + 1
            # amplitudeの値をそのまま使用
            if amplitude >= 0:
                amplitude_str = f"{amplitude:.2f}"
            else:
                amplitude_str = f"m{abs(amplitude):.2f}"
            suggested_name = f"{base_name}_{amplitude_str}_mode{mode_num}.pdb"
            
            # ファイル保存ダイアログ
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Deformed Structure", suggested_name, 
                "PDB files (*.pdb);;All files (*)",
                options=QFileDialog.DontUseNativeDialog)
            
            if file_path:
                # PDBファイルを書き込み
                with open(file_path, 'w') as f:
                    # ヘッダー情報を書き込み
                    f.write(f"REMARK Deformed structure from NMA calculation\n")
                    f.write(f"REMARK Original file: {os.path.basename(self.current_pdb_path)}\n")
                    f.write(f"REMARK Mode: {selected_text}\n")
                    f.write(f"REMARK Amplitude: {amplitude}\n")
                    f.write(f"REMARK Generated by pyNuD NMA\n")
                    f.write(f"REMARK Date: {QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')}\n")
                    
                    # 原子座標を書き込み
                    for i, atom in enumerate(self.nma_selection):
                        x, y, z = new_coords[i]
                        # PDBフォーマットで座標を書き込み
                        line = f"ATOM  {atom.getSerial():5d}  {atom.getName():<3} {atom.getResname():3} {atom.getChid():1}{atom.getResnum():4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.getElement():>2}\n"
                        f.write(line)
                    
                    f.write("END\n")
                
                QMessageBox.information(self, "Success", f"Deformed structure saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save deformed structure:\n{str(e)}")

class SynchronizedInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, main_renderer, secondary_renderer):
        super().__init__()
        self.main_renderer = main_renderer
        self.secondary_renderer = secondary_renderer
        self.syncing = False

        # InteractionEventを監視するオブザーバーを追加
        main_interactor = self.main_renderer.GetRenderWindow().GetInteractor()
        if main_interactor:
            main_interactor.AddObserver(vtk.vtkCommand.InteractionEvent, self.sync_cameras)
    
    # マウスイベントのオーバーライドは不要
    # VTKのカメラオブザーバーが同期をトリガーする

    def sync_cameras(self, caller, event):
        """カメラの位置と向きを同期"""
        if self.syncing:
            return
            
        self.syncing = True
        
        try:
            main_camera = self.main_renderer.GetActiveCamera()
            secondary_camera = self.secondary_renderer.GetActiveCamera()
            
            # カメラの位置、焦点、上方向、視野角を同期
            secondary_camera.SetPosition(main_camera.GetPosition())
            secondary_camera.SetFocalPoint(main_camera.GetFocalPoint())
            secondary_camera.SetViewUp(main_camera.GetViewUp())
            secondary_camera.SetViewAngle(main_camera.GetViewAngle())
            
            # クリッピング範囲も同期して、両方のビューでモデルが正しく表示されるようにする
            secondary_camera.SetClippingRange(main_camera.GetClippingRange())
            
            # レンダリングを更新
            self.secondary_renderer.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error during camera synchronization: {e}")
            
        finally:
            self.syncing = False


class NMAHelpContentManager:
    """
    NMAヘルプコンテンツを管理するクラス
    """
    def __init__(self):
        self._initialize_content()

    def set_language(self, lang_code):
        if lang_code in self.content:
            self.current_language = lang_code

    def get_toc_structure(self):
        return self.content[self.current_language]['toc_structure']

    def get_content(self, page_id):
        pages = self.content[self.current_language]['pages']
        page_content = pages.get(page_id, pages['home'])
        return self._wrap_content(page_content)
            
    def get_ui_text(self, key):
        return self.content[self.current_language]['ui_text'].get(key, '')

    def _wrap_content(self, content):
        return f"""
        <html>
        <head>
        {self.STYLES}
        </head>
        <body>
        {content}
        </body>
        </html>
        """

    STYLES = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 20px; line-height: 1.6; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; font-size: 22px; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 25px; font-size: 18px;}
        h3 { color: #e67e22; margin-top: 20px; font-size: 16px;}
        .highlight { background-color: #f0f8ff; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; }
        .warning { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
        .code { background-color: #f5f5f5; padding: 5px; font-family: monospace; border-radius: 3px; }
        ul { margin: 10px 0; padding-left: 20px; }
        li { margin: 5px 0; }
        strong { color: #000; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
    """
    
    def _initialize_content(self):
        self.current_language = 'ja'
        
        # --- 日本語コンテンツ ---
        toc_structure_ja = [
            ("NMAヘルプ", [
                ("はじめに", "home"),
                ("Normal Mode Analysisとは", "nma_theory"),
                ("基本的な使い方", "basic_usage"),
                ("パラメータ設定", "parameters"),
                ("結果の解釈", "interpretation"),
                ("トラブルシューティング", "troubleshooting"),
            ]),
        ]
        
        pages_ja = {
            "home": """
            <h1>Normal Mode Analysis (NMA) ヘルプ</h1>
            <p>このヘルプでは、pyNuD NMAツールの使い方について詳しく説明します。</p>
            <p>左側の目次から項目を選択して、詳細な説明をご覧ください。</p>
            <div class="highlight">
            <p><strong>Normal Mode Analysis (NMA)</strong>は、タンパク質やその他の生体分子の動的性質を解析するための計算手法です。</p>
            </div>
            """,
            
            "nma_theory": """
            <h1>Normal Mode Analysisとは</h1>
            
            <h2>基本原理</h2>
            <p><strong>Normal Mode Analysis (NMA)</strong>は、タンパク質やその他の生体分子の動的性質を解析するための計算手法です。この手法では、分子の平衡状態周辺での小さな振動を調べることで、分子の柔軟性や機能的な動きを理解することができます。</p>
            
            <h3>NMAの基本原理</h3>
            <ul>
            <li><strong>調和振動子近似</strong>: 分子を調和振動子として扱い、各原子の動きを線形結合で表現</li>
            <li><strong>固有振動モード</strong>: 分子の集団的な動きを表す固有ベクトル（モード）を計算</li>
            <li><strong>固有値</strong>: 各モードの振動周波数に対応する固有値を計算</li>
            <li><strong>低周波モード</strong>: 最も重要な機能的な動きは低周波数（小さな固有値）のモードに含まれる</li>
            </ul>
            
            <h3>NMAの利点</h3>
            <ul>
            <li><strong>計算効率</strong>: 分子動力学シミュレーションと比べて計算コストが低い</li>
            <li><strong>集団的動き</strong>: 分子全体の協調的な動きを捉える</li>
            <li><strong>機能的洞察</strong>: タンパク質の機能に関連する動きを予測</li>
            <li><strong>構造解析</strong>: 分子の柔軟性や安定性を評価</li>
            </ul>
            """,
            
            "basic_usage": """
            <h1>基本的な使い方</h1>
            
            <h2>1. PDBファイルの読み込み</h2>
            <ol>
            <li><strong>Import PDB File...</strong>ボタンをクリック</li>
            <li>解析したいタンパク質のPDBファイルを選択</li>
            <li>ファイルが正常に読み込まれると、左側のビューアに構造が表示されます</li>
            </ol>
            
            <h2>2. 表示設定の調整</h2>
            <ul>
            <li><strong>Style</strong>: 分子の表示スタイルを選択
                <ul>
                <li>Ball & Stick: 原子を球、結合を棒で表示</li>
                <li>Stick Only: 結合のみを表示</li>
                <li>Spheres: 原子を球で表示</li>
                <li>Points: 原子を点で表示</li>
                </ul>
            </li>
            <li><strong>Color</strong>: 色付け方法を選択
                <ul>
                <li>By Element: 元素ごとに色分け</li>
                <li>By Chain: チェーンごとに色分け</li>
                <li>Single Color: 単一色で表示</li>
                <li>By B-Factor: B因子値に基づいて色分け</li>
                </ul>
            </li>
            <li><strong>Show</strong>: 表示する原子を選択
                <ul>
                <li>All Atoms: 全ての原子</li>
                <li>Heavy Atoms: 水素以外の原子</li>
                <li>Backbone: 主鎖原子（N, CA, C）</li>
                <li>C, N, O: 特定の元素のみ</li>
                </ul>
            </li>
            </ul>
            
            <h2>3. NMAシミュレーションの実行</h2>
            <ol>
            <li>パラメータを設定後、<strong>Run NMA Simulation</strong>ボタンをクリック</li>
            <li>計算が進行中はプログレスバーで進捗を確認</li>
            <li>計算完了後、モード選択ドロップダウンに結果が表示されます</li>
            </ol>
            
            <h2>4. 結果の可視化</h2>
            <ul>
            <li><strong>Display Mode</strong>: 表示したいモードを選択</li>
            <li><strong>Amplitude</strong>: 変形の振幅を調整（-100から100）</li>
            </ul>
            
            <h2>5. 変形構造の保存</h2>
            <ol>
            <li>表示したいモードと振幅を設定</li>
            <li><strong>Save Deformed Structure</strong>ボタンをクリック</li>
            <li>保存先ファイル名を指定</li>
            </ol>
            """,
            
            "parameters": """
            <h1>パラメータ設定</h1>
            
            <h2>NMA Parameters</h2>
            
            <h3>Atom Selection（原子選択）</h3>
            <ul>
            <li><strong>calpha</strong>: Cα原子のみ（推奨、高速）
                <ul>
                <li>計算が高速</li>
                <li>タンパク質の主要な動きを捉える</li>
                <li>大きなタンパク質に適している</li>
                </ul>
            </li>
            <li><strong>all</strong>: 全ての原子（高精度、低速）
                <ul>
                <li>より詳細な動きを解析</li>
                <li>計算時間が長い</li>
                <li>小さなタンパク質に適している</li>
                </ul>
            </li>
            </ul>
            
            <h3>Cutoff Distance（カットオフ距離）</h3>
            <p>原子間相互作用のカットオフ距離（5-30 Å）</p>
            <ul>
            <li><strong>小さい値（5-10 Å）</strong>: 局所的な動きに焦点</li>
            <li><strong>推奨値（15 Å）</strong>: バランスの取れた解析</li>
            <li><strong>大きい値（20-30 Å）</strong>: より長距離の相互作用を考慮</li>
            </ul>
            
            <h3>Number of Modes（モード数）</h3>
            <p>計算するモード数（1-100）</p>
            <ul>
            <li><strong>20-50モード</strong>: 通常の解析に十分</li>
            <li><strong>より多くのモード</strong>: 詳細な解析</li>
            <li><strong>少ないモード</strong>: 主要な動きのみ</li>
            </ul>
            
            <div class="warning">
            <strong>注意:</strong> NMA計算にはProDyライブラリが必要です。
            <div class="code">pip install prody</div>
            </div>
            """,
            
            "interpretation": """
            <h1>結果の解釈</h1>
            
            <h2>モードの種類</h2>
            
            <h3>低周波モード（Mode +1, +2, ...）</h3>
            <p>最も重要な機能的な動きを含む</p>
            <ul>
            <li>ドメイン間の相対運動</li>
            <li>リガンド結合部位の動き</li>
            <li>全体的な構造変化</li>
            <li>機能に関連する協調運動</li>
            </ul>
            
            <h3>高周波モード</h3>
            <ul>
            <li>局所的な振動</li>
            <li>熱揺らぎ</li>
            <li>側鎖の動き</li>
            </ul>
            
            <h3>負の固有値モード（Mode -1, -2, ...）</h3>
            <ul>
            <li>構造の不安定性を示す</li>
            <li>変形しやすい方向</li>
            <li>注意深く解釈が必要</li>
            </ul>
            
            <h2>振幅の調整</h2>
            <ul>
            <li><strong>正の値</strong>: 変形の方向</li>
            <li><strong>負の値</strong>: 変形の逆方向</li>
            <li><strong>絶対値</strong>: 変形の大きさ</li>
            </ul>
            
            <h2>結果の評価</h2>
            <ul>
            <li>低周波モードの動きが生物学的に意味があるか</li>
            <li>既知の機能と一致するか</li>
            <li>実験結果と比較</li>
            </ul>
            """,
            
            "troubleshooting": """
            <h1>トラブルシューティング</h1>
            
            <h2>よくある問題と解決方法</h2>
            
            <h3>ProDyがインストールされていない</h3>
            <div class="code">pip install prody</div>
            
            <h3>PDBファイルが読み込めない</h3>
            <ul>
            <li>ファイル形式が正しいか確認</li>
            <li>ファイルが破損していないか確認</li>
            <li>ATOMレコードが含まれているか確認</li>
            <li>ファイルサイズが大きすぎないか確認</li>
            </ul>
            
            <h3>NMA計算が失敗する</h3>
            <ul>
            <li>原子数が多すぎる場合、calpha選択を試す</li>
            <li>カットオフ距離を調整する</li>
            <li>計算するモード数を減らす</li>
            <li>メモリ不足の場合は、より小さなタンパク質を試す</li>
            </ul>
            
            <h3>表示が更新されない</h3>
            <ul>
            <li>モード選択を一度"None"にしてから再選択</li>
            <li>振幅スライダーを動かす</li>
            <li>アプリケーションを再起動</li>
            </ul>
            
            <h3>計算が遅い</h3>
            <ul>
            <li>calpha選択を使用</li>
            <li>モード数を減らす</li>
            <li>カットオフ距離を小さくする</li>
            </ul>
            """,
            
        }
        
        ui_text_ja = {
            'window_title': 'NMA (Normal Mode Analysis) ヘルプ',
            'home_tooltip': 'ホームページに戻る'
        }
        
        # --- 英語コンテンツ ---
        toc_structure_en = [
            ("NMA Help", [
                ("Introduction", "home"),
                ("What is NMA", "nma_theory"),
                ("Basic Usage", "basic_usage"),
                ("Parameter Settings", "parameters"),
                ("Result Interpretation", "interpretation"),
                ("Troubleshooting", "troubleshooting"),
            ]),
        ]
        
        pages_en = {
            "home": """
            <h1>Normal Mode Analysis (NMA) Help</h1>
            <p>This help explains how to use the pyNuD NMA tool in detail.</p>
            <p>Select items from the table of contents on the left to view detailed explanations.</p>
            <div class="highlight">
            <p><strong>Normal Mode Analysis (NMA)</strong> is a computational method for analyzing the dynamic properties of proteins and other biomolecules.</p>
            </div>
            """,
            
            "nma_theory": """
            <h1>What is Normal Mode Analysis</h1>
            
            <h2>Basic Principles</h2>
            <p><strong>Normal Mode Analysis (NMA)</strong> is a computational method for analyzing the dynamic properties of proteins and other biomolecules. This method examines small vibrations around the equilibrium state of molecules to understand their flexibility and functional movements.</p>
            
            <h3>Basic Principles of NMA</h3>
            <ul>
            <li><strong>Harmonic Oscillator Approximation</strong>: Treats molecules as harmonic oscillators, expressing atomic movements as linear combinations</li>
            <li><strong>Eigenvibration Modes</strong>: Calculates eigenvectors (modes) representing collective molecular movements</li>
            <li><strong>Eigenvalues</strong>: Calculates eigenvalues corresponding to vibration frequencies of each mode</li>
            <li><strong>Low-frequency Modes</strong>: Most important functional movements are contained in low-frequency (small eigenvalue) modes</li>
            </ul>
            
            <h3>Advantages of NMA</h3>
            <ul>
            <li><strong>Computational Efficiency</strong>: Lower computational cost compared to molecular dynamics simulations</li>
            <li><strong>Collective Movements</strong>: Captures cooperative movements of entire molecules</li>
            <li><strong>Functional Insights</strong>: Predicts movements related to protein function</li>
            <li><strong>Structural Analysis</strong>: Evaluates molecular flexibility and stability</li>
            </ul>
            """,
            
            "basic_usage": """
            <h1>Basic Usage</h1>
            
            <h2>1. Loading PDB Files</h2>
            <ol>
            <li>Click the <strong>Import PDB File...</strong> button</li>
            <li>Select the PDB file of the protein you want to analyze</li>
            <li>When the file is loaded successfully, the structure will be displayed in the left viewer</li>
            </ol>
            
            <h2>2. Adjusting Display Settings</h2>
            <ul>
            <li><strong>Style</strong>: Select the display style for the molecule
                <ul>
                <li>Ball & Stick: Display atoms as spheres and bonds as sticks</li>
                <li>Stick Only: Display only bonds</li>
                <li>Spheres: Display atoms as spheres</li>
                <li>Points: Display atoms as points</li>
                </ul>
            </li>
            <li><strong>Color</strong>: Select the coloring scheme
                <ul>
                <li>By Element: Color by element type</li>
                <li>By Chain: Color by chain</li>
                <li>Single Color: Single color display</li>
                <li>By B-Factor: Color by B-factor values</li>
                </ul>
            </li>
            <li><strong>Show</strong>: Select which atoms to display
                <ul>
                <li>All Atoms: All atoms</li>
                <li>Heavy Atoms: Non-hydrogen atoms</li>
                <li>Backbone: Backbone atoms (N, CA, C)</li>
                <li>C, N, O: Specific elements only</li>
                </ul>
            </li>
            </ul>
            
            <h2>3. Running NMA Simulation</h2>
            <ol>
            <li>After setting parameters, click the <strong>Run NMA Simulation</strong> button</li>
            <li>Check progress with the progress bar during calculation</li>
            <li>After calculation is complete, results will be displayed in the mode selection dropdown</li>
            </ol>
            
            <h2>4. Visualizing Results</h2>
            <ul>
            <li><strong>Display Mode</strong>: Select the mode you want to display</li>
            <li><strong>Amplitude</strong>: Adjust the amplitude of deformation (-100 to 100)</li>
            </ul>
            
            <h2>5. Saving Deformed Structures</h2>
            <ol>
            <li>Set the desired mode and amplitude</li>
            <li>Click the <strong>Save Deformed Structure</strong> button</li>
            <li>Specify the output filename</li>
            </ol>
            """,
            
            "parameters": """
            <h1>Parameter Settings</h1>
            
            <h2>NMA Parameters</h2>
            
            <h3>Atom Selection</h3>
            <ul>
            <li><strong>calpha</strong>: Cα atoms only (recommended, fast)
                <ul>
                <li>Fast calculation</li>
                <li>Captures major protein movements</li>
                <li>Suitable for large proteins</li>
                </ul>
            </li>
            <li><strong>all</strong>: All atoms (high precision, slow)
                <ul>
                <li>More detailed movement analysis</li>
                <li>Longer calculation time</li>
                <li>Suitable for small proteins</li>
                </ul>
            </li>
            </ul>
            
            <h3>Cutoff Distance</h3>
            <p>Cutoff distance for interatomic interactions (5-30 Å)</p>
            <ul>
            <li><strong>Small values (5-10 Å)</strong>: Focus on local movements</li>
            <li><strong>Recommended value (15 Å)</strong>: Balanced analysis</li>
            <li><strong>Large values (20-30 Å)</strong>: Consider longer-range interactions</li>
            </ul>
            
            <h3>Number of Modes</h3>
            <p>Number of modes to calculate (1-100)</p>
            <ul>
            <li><strong>20-50 modes</strong>: Sufficient for normal analysis</li>
            <li><strong>More modes</strong>: Detailed analysis</li>
            <li><strong>Fewer modes</strong>: Major movements only</li>
            </ul>
            
            <div class="warning">
            <strong>Note:</strong> ProDy library is required for NMA calculation.
            <div class="code">pip install prody</div>
            </div>
            """,
            
            "interpretation": """
            <h1>Result Interpretation</h1>
            
            <h2>Types of Modes</h2>
            
            <h3>Low-frequency Modes (Mode +1, +2, ...)</h3>
            <p>Contain the most important functional movements</p>
            <ul>
            <li>Inter-domain relative movements</li>
            <li>Ligand binding site movements</li>
            <li>Overall structural changes</li>
            <li>Function-related cooperative movements</li>
            </ul>
            
            <h3>High-frequency Modes</h3>
            <ul>
            <li>Local vibrations</li>
            <li>Thermal fluctuations</li>
            <li>Side chain movements</li>
            </ul>
            
            <h3>Negative Eigenvalue Modes (Mode -1, -2, ...)</h3>
            <ul>
            <li>Indicate structural instability</li>
            <li>Directions prone to deformation</li>
            <li>Require careful interpretation</li>
            </ul>
            
            <h2>Amplitude Adjustment</h2>
            <ul>
            <li><strong>Positive values</strong>: Direction of deformation</li>
            <li><strong>Negative values</strong>: Opposite direction of deformation</li>
            <li><strong>Absolute values</strong>: Magnitude of deformation</li>
            </ul>
            
            <h2>Result Evaluation</h2>
            <ul>
            <li>Whether low-frequency mode movements are biologically meaningful</li>
            <li>Consistency with known functions</li>
            <li>Comparison with experimental results</li>
            </ul>
            """,
            
            "troubleshooting": """
            <h1>Troubleshooting</h1>
            
            <h2>Common Problems and Solutions</h2>
            
            <h3>ProDy is not installed</h3>
            <div class="code">pip install prody</div>
            
            <h3>Cannot load PDB file</h3>
            <ul>
            <li>Check if the file format is correct</li>
            <li>Check if the file is corrupted</li>
            <li>Check if ATOM records are included</li>
            <li>Check if the file size is not too large</li>
            </ul>
            
            <h3>NMA calculation fails</h3>
            <ul>
            <li>Try calpha selection if there are too many atoms</li>
            <li>Adjust the cutoff distance</li>
            <li>Reduce the number of modes to calculate</li>
            <li>Try smaller proteins if memory is insufficient</li>
            </ul>
            
            <h3>Display does not update</h3>
            <ul>
            <li>Set mode selection to "None" once and then reselect</li>
            <li>Move the amplitude slider</li>
            <li>Restart the application</li>
            </ul>
            
            <h3>Calculation is slow</h3>
            <ul>
            <li>Use calpha selection</li>
            <li>Reduce the number of modes</li>
            <li>Decrease the cutoff distance</li>
            </ul>
            """,
            
        }
        
        ui_text_en = {
            'window_title': 'NMA (Normal Mode Analysis) Help',
            'home_tooltip': 'Return to home page'
        }
        
        self.content = {
            'ja': {
                'toc_structure': toc_structure_ja,
                'pages': pages_ja,
                'ui_text': ui_text_ja
            },
            'en': {
                'toc_structure': toc_structure_en,
                'pages': pages_en,
                'ui_text': ui_text_en
            }
        }


class NMAHelpWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NMA (Normal Mode Analysis) ヘルプ")
        self.setGeometry(100, 100, 900, 700)
        
        # 独立したウィンドウとして設定
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        self.content_manager = NMAHelpContentManager()
        self.setupUI()

    def setupUI(self):
        # 中央ウィジェットを作成
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # ツールバー
        toolbar = QToolBar()
        home_action = QAction("🏠", self)
        home_action.setToolTip("ホームページに戻る")
        home_action.triggered.connect(self.showHomePage)
        toolbar.addAction(home_action)
        
        # 言語切り替えボタン
        toolbar.addSeparator()
        self.ja_action = QAction("🇯🇵 日本語", self)
        self.ja_action.triggered.connect(lambda: self.switch_language('ja'))
        toolbar.addAction(self.ja_action)
        
        self.en_action = QAction("🇺🇸 English", self)
        self.en_action.triggered.connect(lambda: self.switch_language('en'))
        toolbar.addAction(self.en_action)
        
        self.addToolBar(toolbar)
        
        # スプリッター
        splitter = QSplitter(Qt.Horizontal)
        
        # 目次ツリー
        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderLabel("目次")
        self.toc_tree.setMaximumWidth(250)
        self.toc_tree.itemClicked.connect(self.onTocItemClicked)
        
        # ヘルプビューア
        self.help_viewer = QTextEdit()
        self.help_viewer.setReadOnly(True)
        
        splitter.addWidget(self.toc_tree)
        splitter.addWidget(self.help_viewer)
        
        splitter.setSizes([220, 680])
        layout.addWidget(splitter)
        
        self.loadTocContent()
        self.showHomePage()

    def switch_language(self, lang_code):
        self.content_manager.set_language(lang_code)
        self.setWindowTitle(self.content_manager.get_ui_text('window_title'))
        self.loadTocContent()
        self.showHomePage()

    def loadTocContent(self):
        self.toc_tree.clear()
        toc_structure = self.content_manager.get_toc_structure()
        
        def add_items(parent_item, items_list):
            for item_data in items_list:
                name, item_id = item_data
                child_item = QTreeWidgetItem([name])
                child_item.setData(0, Qt.UserRole, item_id)
                parent_item.addChild(child_item)
        
        for category_name, items in toc_structure:
            category_item = QTreeWidgetItem([category_name])
            self.toc_tree.addTopLevelItem(category_item)
            add_items(category_item, items)
        
        self.toc_tree.expandAll()
    
    def onTocItemClicked(self, item, column):
        item_id = item.data(0, Qt.UserRole)
        if item_id:
            self.showHelpPage(item_id)
    
    def showHelpPage(self, page_id):
        self.help_viewer.setHtml(self.content_manager.get_content(page_id))
    
    def showHomePage(self):
        self.showHelpPage('home')


def create_plugin(main_window):
    """Plugin entry point. Called from pyNuD Plugin menu."""
    return NMA_Window(main_window=main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "NMA_Window"]


if __name__ == "__main__":
    # アプリケーション作成前にHighDPI設定
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    window = NMA_Window()
    window.show()
    sys.exit(app.exec_())
