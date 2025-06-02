import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QSizePolicy,
    QAction,
    QMenuBar,
    QStatusBar,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QSlider,
    QCheckBox,
    QProgressBar,
    QSplitter,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QCursor, QFont
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QThread, pyqtSignal, QTimer
from segmentation import run_grabcut_logic, apply_border_matting_logic


class ProcessingThread(QThread):
    """后台处理线程，避免界面卡顿"""

    finished = pyqtSignal(object, object, str)  # mask, binary, message

    def __init__(
        self,
        original_image,
        grabcut_mask,
        roi_tuple,
        fg_scribbles,
        bg_scribbles,
        brush_size,
        mode,
    ):
        super().__init__()
        self.original_image = original_image
        self.grabcut_mask = grabcut_mask
        self.roi_tuple = roi_tuple
        self.fg_scribbles = fg_scribbles
        self.bg_scribbles = bg_scribbles
        self.brush_size = brush_size
        self.mode = mode

    def run(self):
        updated_mask, output_binary, status_msg = run_grabcut_logic(
            self.original_image,
            self.grabcut_mask,
            self.roi_tuple,
            self.fg_scribbles,
            self.bg_scribbles,
            self.brush_size,
            self.mode,
        )
        self.finished.emit(updated_mask, output_binary, status_msg)


class ImageLabel(QLabel):
    """改进的图像显示组件"""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setMinimumSize(600, 400)  # 增大最小尺寸
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("""
            ImageLabel {
                border: 2px solid #d0d0d0;
                border-radius: 8px;
                background-color: #fafafa;
            }
        """)

        # 状态变量
        self.current_roi_rect_label = QRect()
        self.drawing_roi = False
        self.scribbles_fg = []
        self.scribbles_bg = []
        self.current_scribble_img_coords = []
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.last_pan_point = QPoint()
        self.is_panning = False

        self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        """添加缩放功能"""
        if self.parent_app.original_image is None:
            return

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.zoom_factor *= zoom_factor
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))  # 限制缩放范围

        self.parent_app.update_display()

    def mousePressEvent(self, event):
        if self.parent_app.original_image is None:
            return

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier
        ):
            # 平移模式
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # ...existing code...
        pos_in_label = event.pos()
        img_coord = self.map_label_to_image_coords(pos_in_label)

        if not img_coord:
            return

        if self.parent_app.interaction_mode == "roi":
            if event.button() == Qt.LeftButton:
                self.parent_app.save_state_for_undo()
                self.drawing_roi = True
                self.current_roi_rect_label.setTopLeft(pos_in_label)
                self.current_roi_rect_label.setBottomRight(pos_in_label)
        elif self.parent_app.interaction_mode in ["fg_scribble", "bg_scribble"]:
            if event.button() == Qt.LeftButton:
                self.parent_app.save_state_for_undo()
                self.current_scribble_img_coords = [img_coord]

        self.update()

    # ...existing mouseMoveEvent and mouseReleaseEvent methods...

    def clear_interactions(self):
        """清除所有交互标记"""
        self.current_roi_rect_label = QRect()
        self.scribbles_fg.clear()
        self.scribbles_bg.clear()
        self.current_scribble_img_coords = []
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()


class ControlPanel(QWidget):
    """独立的控制面板组件"""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.init_ui()

    def init_ui(self):
        self.setFixedWidth(250)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # 交互模式组
        mode_group = QGroupBox("交互模式")
        mode_layout = QVBoxLayout()

        self.mode_button_group = QButtonGroup(self)
        self.roi_mode_button = QRadioButton("🎯 框选ROI")
        self.roi_mode_button.setChecked(True)
        self.fg_scribble_button = QRadioButton("🖌️ 前景涂鸦")
        self.bg_scribble_button = QRadioButton("🗑️ 背景涂鸦")

        for button in [
            self.roi_mode_button,
            self.fg_scribble_button,
            self.bg_scribble_button,
        ]:
            mode_layout.addWidget(button)
            self.mode_button_group.addButton(button)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 工具设置组
        tools_group = QGroupBox("工具设置")
        tools_layout = QVBoxLayout()

        # 笔刷大小
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("笔刷大小:"))
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 50)
        self.brush_size_spinbox.setValue(5)
        brush_layout.addWidget(self.brush_size_spinbox)
        tools_layout.addLayout(brush_layout)

        # 透明度设置
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("结果透明度:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_label = QLabel("80%")
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        tools_layout.addLayout(opacity_layout)

        # 显示选项
        self.show_mask_overlay = QCheckBox("显示掩码叠加")
        self.show_mask_overlay.setChecked(True)
        tools_layout.addWidget(self.show_mask_overlay)

        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # 操作按钮组
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout()

        # 主要操作按钮
        self.run_grabcut_button = QPushButton("🎯 运行分割")
        self.run_grabcut_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; }"
        )

        self.border_matting_button = QPushButton("✨ 边缘优化")
        self.border_matting_button.setStyleSheet(
            "QPushButton { background-color: #FF9800; }"
        )

        # 辅助操作按钮
        self.undo_button = QPushButton("↶ 撤销")
        self.clear_scribbles_button = QPushButton("🧹 清除涂鸦")
        self.reset_current_button = QPushButton("🔄 重置当前")
        self.reset_button = QPushButton("🗑️ 重置所有")

        for button in [
            self.run_grabcut_button,
            self.border_matting_button,
            self.undo_button,
            self.clear_scribbles_button,
            self.reset_current_button,
            self.reset_button,
        ]:
            actions_layout.addWidget(button)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # 连接信号
        self.opacity_slider.valueChanged.connect(self.update_opacity_label)

    def update_opacity_label(self, value):
        self.opacity_label.setText(f"{value}%")


class InteractiveSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 交互式图像分割系统")
        self.setGeometry(100, 100, 1400, 900)

        # 数据属性
        self.original_image = None
        self.display_image_cv = None
        self.display_pixmap = None
        self.grabcut_mask = None
        self.current_image_roi_qrect = None

        # 交互属性
        self.interaction_mode = "roi"
        self.fg_color = QColor(0, 255, 0, 180)
        self.bg_color = QColor(255, 0, 0, 180)
        self.brush_size = 5

        # 撤销功能
        self.undo_stack = []
        self.max_undo_steps = 20  # 限制撤销步数

        # 处理线程
        self.processing_thread = None

        self.init_ui()
        self.apply_modern_theme()

        # 添加状态栏定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_status_message)

    def init_ui(self):
        # 创建菜单栏
        self.create_menu_bar()

        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 使用分割器布局
        splitter = QSplitter(Qt.Horizontal)

        # 控制面板
        self.control_panel = ControlPanel(self)
        splitter.addWidget(self.control_panel)

        # 图像显示区域
        self.image_label = ImageLabel(self)
        splitter.addWidget(self.image_label)

        # 设置分割器比例
        splitter.setSizes([250, 1000])
        splitter.setCollapsible(0, False)  # 控制面板不可折叠

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("请先加载图像 📸")

        self.coord_label = QLabel("")
        self.coord_label.setMinimumWidth(200)
        self.status_bar.addPermanentWidget(self.coord_label)

        # 连接信号
        self.connect_signals()

        # 初始状态
        self.set_interaction_mode("roi")
        self.update_ui_state()

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("📁 文件")

        load_action = QAction("📂 加载图像...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        save_action = QAction("💾 保存结果...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        recent_menu = file_menu.addMenu("📋 最近文件")
        # TODO: 实现最近文件功能

        file_menu.addSeparator()

        exit_action = QAction("🚪 退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu("✏️ 编辑")

        undo_action = QAction("↶ 撤销", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)

        clear_action = QAction("🧹 清除涂鸦", self)
        clear_action.triggered.connect(self.clear_scribbles)
        edit_menu.addAction(clear_action)

        # 视图菜单
        view_menu = menubar.addMenu("👁️ 视图")

        zoom_in_action = QAction("🔍 放大", self)
        zoom_in_action.setShortcut("Ctrl++")
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("🔍 缩小", self)
        zoom_out_action.setShortcut("Ctrl+-")
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("📐 适应窗口", self)
        fit_action.setShortcut("Ctrl+0")
        view_menu.addAction(fit_action)

    def connect_signals(self):
        """连接所有信号"""
        # 模式切换
        self.control_panel.roi_mode_button.toggled.connect(
            lambda: self.set_interaction_mode("roi")
        )
        self.control_panel.fg_scribble_button.toggled.connect(
            lambda: self.set_interaction_mode("fg_scribble")
        )
        self.control_panel.bg_scribble_button.toggled.connect(
            lambda: self.set_interaction_mode("bg_scribble")
        )

        # 工具设置
        self.control_panel.brush_size_spinbox.valueChanged.connect(self.set_brush_size)
        self.control_panel.opacity_slider.valueChanged.connect(self.update_display)
        self.control_panel.show_mask_overlay.toggled.connect(self.update_display)

        # 操作按钮
        self.control_panel.run_grabcut_button.clicked.connect(self.manual_run_grabcut)
        self.control_panel.border_matting_button.clicked.connect(
            self.apply_border_matting
        )
        self.control_panel.undo_button.clicked.connect(self.undo_action)
        self.control_panel.clear_scribbles_button.clicked.connect(self.clear_scribbles)
        self.control_panel.reset_current_button.clicked.connect(
            self.reset_current_image_state
        )
        self.control_panel.reset_button.clicked.connect(self.reset_all)

    def apply_modern_theme(self):
        """应用现代化主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            
            QLabel {
                font-size: 11pt;
                color: #2c3e50;
            }
            
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 16px;
                text-align: center;
                font-size: 11pt;
                font-weight: bold;
                margin: 2px;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background-color: #45a049;
                transform: translateY(-1px);
            }
            
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            
            QRadioButton {
                font-size: 11pt;
                padding: 5px;
                color: #2c3e50;
            }
            
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            
            QSpinBox, QSlider {
                padding: 4px;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                font-size: 11pt;
                background-color: white;
            }
            
            QSpinBox:focus, QSlider:focus {
                border-color: #3498db;
            }
            
            QStatusBar {
                background-color: #ecf0f1;
                border-top: 1px solid #bdc3c7;
                font-size: 10pt;
            }
            
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)

    def manual_run_grabcut(self):
        """手动运行GrabCut"""
        if self.original_image is None:
            self.show_status_message("请先加载图像 📸", 3000)
            return

        # 根据当前状态决定运行模式
        if self.current_image_roi_qrect:
            self.run_grabcut(init_with_rect=True)
        elif self.image_label.scribbles_fg or self.image_label.scribbles_bg:
            self.run_grabcut(init_with_mask=True)
        else:
            self.show_status_message("请先框选ROI或添加涂鸦标记 🎯", 3000)

    def clear_scribbles(self):
        """清除所有涂鸦"""
        if not self.image_label.scribbles_fg and not self.image_label.scribbles_bg:
            self.show_status_message("没有涂鸦需要清除", 2000)
            return

        self.save_state_for_undo()
        self.image_label.scribbles_fg.clear()
        self.image_label.scribbles_bg.clear()
        self.image_label.update()
        self.show_status_message("涂鸦已清除 🧹", 2000)

    def update_ui_state(self):
        """更新UI状态"""
        has_image = self.original_image is not None
        has_result = self.grabcut_mask is not None and np.any(self.grabcut_mask > 0)

        # 更新按钮状态
        self.control_panel.run_grabcut_button.setEnabled(has_image)
        self.control_panel.border_matting_button.setEnabled(has_result)
        self.control_panel.undo_button.setEnabled(len(self.undo_stack) > 1)
        self.control_panel.clear_scribbles_button.setEnabled(
            bool(self.image_label.scribbles_fg or self.image_label.scribbles_bg)
        )

    def show_status_message(self, message, timeout=0):
        """显示状态消息"""
        self.status_bar.showMessage(message)
        if timeout > 0:
            self.status_timer.start(timeout)

    def clear_status_message(self):
        """清除状态消息"""
        self.status_timer.stop()
        self.status_bar.clearMessage()

    def save_state_for_undo(self):
        """保存状态用于撤销"""
        if self.original_image is None:
            return

        # 限制撤销栈大小
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)

        state = {
            "display_image_cv": self.display_image_cv.copy()
            if self.display_image_cv is not None
            else None,
            "grabcut_mask": self.grabcut_mask.copy()
            if self.grabcut_mask is not None
            else None,
            "current_image_roi_qrect": self.current_image_roi_qrect.normalized()
            if self.current_image_roi_qrect
            else None,
            "scribbles_fg": [list(path) for path in self.image_label.scribbles_fg],
            "scribbles_bg": [list(path) for path in self.image_label.scribbles_bg],
        }
        self.undo_stack.append(state)
        self.update_ui_state()

    def run_grabcut(self, init_with_rect=False, init_with_mask=False):
        """运行GrabCut算法"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.show_status_message("处理中，请稍候... ⏳", 1000)
            return

        # 显示进度条
        self.control_panel.progress_bar.setVisible(True)
        self.control_panel.progress_bar.setRange(0, 0)  # 无限进度条

        # 准备参数
        roi_tuple_for_logic = None
        if self.current_image_roi_qrect:
            r = self.current_image_roi_qrect
            roi_tuple_for_logic = (r.x(), r.y(), r.width(), r.height())

        fg_scribbles_points = [
            [(p.x(), p.y()) for p in path] for path in self.image_label.scribbles_fg
        ]
        bg_scribbles_points = [
            [(p.x(), p.y()) for p in path] for path in self.image_label.scribbles_bg
        ]

        # 确定运行模式
        if init_with_rect and roi_tuple_for_logic:
            mode = "INIT_WITH_RECT"
        elif init_with_mask or fg_scribbles_points or bg_scribbles_points:
            mode = "INIT_WITH_MASK"
        else:
            mode = "INIT_WITH_RECT"

        # 启动后台处理
        self.processing_thread = ProcessingThread(
            self.original_image,
            self.grabcut_mask.copy() if self.grabcut_mask is not None else None,
            roi_tuple_for_logic,
            fg_scribbles_points,
            bg_scribbles_points,
            self.brush_size,
            mode,
        )
        self.processing_thread.finished.connect(self.on_grabcut_finished)
        self.processing_thread.start()

        self.show_status_message("正在处理分割... 🎯")

    def on_grabcut_finished(self, updated_mask, output_binary, status_msg):
        """GrabCut处理完成回调"""
        self.control_panel.progress_bar.setVisible(False)

        if updated_mask is not None and output_binary is not None:
            self.grabcut_mask = updated_mask

            # 根据设置应用透明度
            opacity = self.control_panel.opacity_slider.value() / 100.0
            if self.control_panel.show_mask_overlay.isChecked():
                # 显示带透明度的结果
                alpha = (output_binary * 255 * opacity).astype(np.uint8)
                self.display_image_cv = np.dstack(
                    [
                        self.original_image[:, :, 0],
                        self.original_image[:, :, 1],
                        self.original_image[:, :, 2],
                        alpha,
                    ]
                )
            else:
                self.display_image_cv = (
                    self.original_image * output_binary[:, :, np.newaxis]
                )

            self.update_display()
            self.save_state_for_undo()
            self.update_ui_state()

        self.show_status_message(status_msg, 3000)

    # ...existing methods for load_image, save_result, etc...
    # (保持原有的核心逻辑方法，但可以添加更好的错误处理和用户反馈)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("交互式图像分割系统")
    app.setApplicationVersion("2.0")

    main_window = InteractiveSegmentationApp()
    main_window.show()

    sys.exit(app.exec_())
