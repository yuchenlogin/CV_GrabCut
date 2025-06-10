import sys
import cv2
import numpy as np
import time
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
    QFrame,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QCursor, QFont
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QThread, pyqtSignal, QTimer
from segmentation import run_grabcut_logic, apply_border_matting_logic


class ProcessingThread(QThread):
    """后台处理线程，避免界面卡顿"""

    finished = pyqtSignal(object, object, str, float)  # mask, binary, message

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
        try:
            start_time = time.perf_counter()  # 使用高精度计时器
            updated_mask, output_binary, status_msg = run_grabcut_logic(
                self.original_image,
                self.grabcut_mask,
                self.roi_tuple,
                self.fg_scribbles,
                self.bg_scribbles,
                self.brush_size,
                self.mode,
            )
            end_time = time.perf_counter()
            execution_time = end_time - start_time  # 计算执行耗时（单位：秒）
            self.finished.emit(updated_mask, output_binary, status_msg, execution_time)
        except Exception as e:
            self.finished.emit(None, None, f"处理出错: {str(e)}")


class ImageLabel(QLabel):
    """自定义QLabel用于显示图像并处理鼠标交互事件"""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setMinimumSize(600, 400)
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

        # 缩放和平移
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.last_pan_point = QPoint()
        self.is_panning = False

        self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if self.parent_app.original_image is None:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.zoom_factor *= zoom_factor
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))

        self.parent_app.update_display()

    def mousePressEvent(self, event):
        if self.parent_app.original_image is None:
            return

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier
        ):
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

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

    def mouseMoveEvent(self, event):
        if self.parent_app.original_image is None:
            return

        if self.is_panning:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.parent_app.update_display()
            return

        pos_in_label = event.pos()
        img_coord = self.map_label_to_image_coords(pos_in_label)
        self.parent_app.update_status_bar_coords(img_coord)

        if self.parent_app.interaction_mode == "roi":
            if self.drawing_roi:
                self.current_roi_rect_label.setBottomRight(pos_in_label)
        elif self.parent_app.interaction_mode in ["fg_scribble", "bg_scribble"]:
            if event.buttons() & Qt.LeftButton and self.current_scribble_img_coords:
                if img_coord:
                    self.current_scribble_img_coords.append(img_coord)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.parent_app.original_image is None:
            return

        if self.is_panning:
            self.is_panning = False
            self.parent_app.set_interaction_mode(self.parent_app.interaction_mode)
            return

        if self.parent_app.interaction_mode == "roi":
            if event.button() == Qt.LeftButton and self.drawing_roi:
                self.drawing_roi = False
                normalized_label_roi = self.current_roi_rect_label.normalized()

                top_left_img = self.map_label_to_image_coords(
                    normalized_label_roi.topLeft()
                )
                bottom_right_img = self.map_label_to_image_coords(
                    normalized_label_roi.bottomRight()
                )

                if top_left_img and bottom_right_img:
                    self.parent_app.current_image_roi_qrect = QRect(
                        top_left_img, bottom_right_img
                    ).normalized()
                    if (
                        self.parent_app.current_image_roi_qrect.width() > 0
                        and self.parent_app.current_image_roi_qrect.height() > 0
                    ):
                        self.parent_app.run_grabcut(init_with_rect=True)
                    else:
                        self.parent_app.current_image_roi_qrect = None
                        self.parent_app.show_status_message(
                            "ROI选择过小或无效 ❌", 3000
                        )
                else:
                    self.parent_app.current_image_roi_qrect = None
                self.current_roi_rect_label = QRect()

        elif self.parent_app.interaction_mode in ["fg_scribble", "bg_scribble"]:
            if event.button() == Qt.LeftButton and self.current_scribble_img_coords:
                if len(self.current_scribble_img_coords) > 1:
                    if self.parent_app.interaction_mode == "fg_scribble":
                        self.scribbles_fg.append(list(self.current_scribble_img_coords))
                    else:
                        self.scribbles_bg.append(list(self.current_scribble_img_coords))
                elif len(self.current_scribble_img_coords) == 1:
                    point = self.current_scribble_img_coords[0]
                    tiny_segment = [point, QPoint(point.x() + 1, point.y() + 1)]
                    if self.parent_app.interaction_mode == "fg_scribble":
                        self.scribbles_fg.append(tiny_segment)
                    else:
                        self.scribbles_bg.append(tiny_segment)

                self.current_scribble_img_coords = []
                self.parent_app.update_ui_state()

                if self.parent_app.grabcut_mask is not None:
                    self.parent_app.run_grabcut(init_with_mask=True)
                elif self.parent_app.original_image is not None:
                    self.parent_app.show_status_message(
                        "建议先框选ROI，或确保GrabCut已基于ROI初始化 🎯", 3000
                    )

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if (
            self.parent_app.display_pixmap is None
            or self.parent_app.display_pixmap.isNull()
        ):
            return

        painter = QPainter(self)

        # 绘制ROI
        if self.drawing_roi and not self.current_roi_rect_label.isNull():
            pen = QPen(QColor(0, 255, 255, 180), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_roi_rect_label.normalized())

        # 绘制前景涂鸦
        pen_fg = QPen(
            self.parent_app.fg_color,
            self.parent_app.brush_size,
            Qt.SolidLine,
            Qt.RoundCap,
            Qt.RoundJoin,
        )
        painter.setPen(pen_fg)
        for scribble_path_img_coords in self.scribbles_fg:
            if len(scribble_path_img_coords) > 1:
                poly_points_label = [
                    self.map_image_to_label_coords(pt)
                    for pt in scribble_path_img_coords
                ]
                poly_points_label = [pt for pt in poly_points_label if pt is not None]
                if len(poly_points_label) > 1:
                    painter.drawPolyline(*poly_points_label)
            elif len(scribble_path_img_coords) == 1:
                pt_label = self.map_image_to_label_coords(scribble_path_img_coords[0])
                if pt_label:
                    painter.drawPoint(pt_label)

        # 绘制背景涂鸦
        pen_bg = QPen(
            self.parent_app.bg_color,
            self.parent_app.brush_size,
            Qt.SolidLine,
            Qt.RoundCap,
            Qt.RoundJoin,
        )
        painter.setPen(pen_bg)
        for scribble_path_img_coords in self.scribbles_bg:
            if len(scribble_path_img_coords) > 1:
                poly_points_label = [
                    self.map_image_to_label_coords(pt)
                    for pt in scribble_path_img_coords
                ]
                poly_points_label = [pt for pt in poly_points_label if pt is not None]
                if len(poly_points_label) > 1:
                    painter.drawPolyline(*poly_points_label)
            elif len(scribble_path_img_coords) == 1:
                pt_label = self.map_image_to_label_coords(scribble_path_img_coords[0])
                if pt_label:
                    painter.drawPoint(pt_label)

        # 绘制当前涂鸦
        if (
            self.current_scribble_img_coords
            and len(self.current_scribble_img_coords) > 1
        ):
            current_pen = (
                pen_fg if self.parent_app.interaction_mode == "fg_scribble" else pen_bg
            )
            painter.setPen(current_pen)
            poly_points_label = [
                self.map_image_to_label_coords(pt)
                for pt in self.current_scribble_img_coords
            ]
            poly_points_label = [pt for pt in poly_points_label if pt is not None]
            if len(poly_points_label) > 1:
                painter.drawPolyline(*poly_points_label)

    def map_label_to_image_coords(self, label_pos):
        if (
            self.parent_app.original_image is None
            or self.parent_app.display_pixmap is None
            or self.parent_app.display_pixmap.isNull()
        ):
            return None

        pixmap_w = self.parent_app.display_pixmap.width()
        pixmap_h = self.parent_app.display_pixmap.height()
        label_w = self.width()
        label_h = self.height()

        offset_x = (label_w - pixmap_w) / 2 + self.pan_offset.x()
        offset_y = (label_h - pixmap_h) / 2 + self.pan_offset.y()

        pixmap_x = (label_pos.x() - offset_x) / self.zoom_factor
        pixmap_y = (label_pos.y() - offset_y) / self.zoom_factor

        if not (0 <= pixmap_x < pixmap_w and 0 <= pixmap_y < pixmap_h):
            return None

        orig_h, orig_w = self.parent_app.original_image.shape[:2]
        scale_x = orig_w / pixmap_w
        scale_y = orig_h / pixmap_h

        img_x = int(pixmap_x * scale_x)
        img_y = int(pixmap_y * scale_y)

        img_x = max(0, min(img_x, orig_w - 1))
        img_y = max(0, min(img_y, orig_h - 1))

        return QPoint(img_x, img_y)

    def map_image_to_label_coords(self, image_pos):
        if (
            self.parent_app.original_image is None
            or self.parent_app.display_pixmap is None
            or self.parent_app.display_pixmap.isNull()
        ):
            return None

        pixmap_w = self.parent_app.display_pixmap.width()
        pixmap_h = self.parent_app.display_pixmap.height()
        label_w = self.width()
        label_h = self.height()

        offset_x = (label_w - pixmap_w) / 2 + self.pan_offset.x()
        offset_y = (label_h - pixmap_h) / 2 + self.pan_offset.y()

        orig_h, orig_w = self.parent_app.original_image.shape[:2]
        if orig_w == 0 or orig_h == 0:
            return None

        scale_x = pixmap_w / orig_w
        scale_y = pixmap_h / orig_h

        pixmap_x_float = image_pos.x() * scale_x * self.zoom_factor
        pixmap_y_float = image_pos.y() * scale_y * self.zoom_factor

        label_x = int(pixmap_x_float + offset_x)
        label_y = int(pixmap_y_float + offset_y)

        return QPoint(label_x, label_y)

    def clear_interactions(self):
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
        self.setFixedWidth(280)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
        """)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # 交互模式组
        mode_group = QGroupBox("🎯 交互模式")
        mode_layout = QVBoxLayout()

        self.mode_button_group = QButtonGroup(self)
        self.roi_mode_button = QRadioButton("📐 框选ROI")
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
        tools_group = QGroupBox("🔧 工具设置")
        tools_layout = QVBoxLayout()

        # 笔刷大小
        brush_frame = QFrame()
        brush_layout = QHBoxLayout(brush_frame)
        brush_layout.setContentsMargins(0, 0, 0, 0)
        brush_layout.addWidget(QLabel("笔刷大小:"))
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 50)
        self.brush_size_spinbox.setValue(5)
        brush_layout.addWidget(self.brush_size_spinbox)
        tools_layout.addWidget(brush_frame)

        # 透明度设置
        opacity_frame = QFrame()
        opacity_layout = QVBoxLayout(opacity_frame)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_label_layout = QHBoxLayout()
        opacity_label_layout.addWidget(QLabel("结果透明度:"))
        self.opacity_label = QLabel("80%")
        opacity_label_layout.addWidget(self.opacity_label)
        opacity_layout.addLayout(opacity_label_layout)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        opacity_layout.addWidget(self.opacity_slider)
        tools_layout.addWidget(opacity_frame)

        # 显示选项
        self.show_mask_overlay = QCheckBox("显示掩码叠加")
        self.show_mask_overlay.setChecked(True)
        tools_layout.addWidget(self.show_mask_overlay)

        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # 操作按钮组
        actions_group = QGroupBox("⚡ 操作")
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
        self.setWindowTitle("🎨 交互式图像分割系统 v2.0")
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
        self.max_undo_steps = 20

        # 处理线程
        self.processing_thread = None

        self.init_ui()
        self.apply_modern_theme()

        # 状态栏定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_status_message)

    def init_ui(self):
        self.create_menu_bar()

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
        splitter.setSizes([280, 1120])
        splitter.setCollapsible(0, False)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("欢迎使用图像分割系统！请先加载图像 📸")

        self.coord_label = QLabel("")
        self.coord_label.setMinimumWidth(200)
        self.status_bar.addPermanentWidget(self.coord_label)

        # 连接信号
        self.connect_signals()

        # 初始状态
        self.set_interaction_mode("roi")
        self.update_ui_state()

    def create_menu_bar(self):
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
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("🔍 缩小", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("📐 适应窗口", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)

    def connect_signals(self):
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
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            
            QLabel {
                font-size: 11pt;
                color: #2c3e50;
                background-color: transparent;
            }
            
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 12px 16px;
                text-align: center;
                font-size: 11pt;
                font-weight: bold;
                margin: 2px;
                border-radius: 8px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background-color: #45a049;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            QPushButton:pressed {
                background-color: #3e8e41;
                transform: translateY(1px);
            }
            
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            
            /* 特殊按钮样式 */
            QPushButton[objectName="run_grabcut_button"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: 2px solid #2980b9;
            }
            
            QPushButton[objectName="run_grabcut_button"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }
            
            QPushButton[objectName="border_matting_button"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
                border: 2px solid #e67e22;
            }
            
            QPushButton[objectName="border_matting_button"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f7dc6f, stop:1 #f39c12);
            }
            
            QPushButton[objectName="undo_button"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9b59b6, stop:1 #8e44ad);
                border: 2px solid #8e44ad;
            }
            
            QPushButton[objectName="reset_button"], 
            QPushButton[objectName="reset_current_button"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
                border: 2px solid #c0392b;
            }
            
            QRadioButton {
                font-size: 11pt;
                padding: 8px;
                color: #2c3e50;
                background-color: transparent;
                spacing: 8px;
            }
            
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid #bdc3c7;
                background-color: white;
            }
            
            QRadioButton::indicator:checked {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
            
            QRadioButton::indicator:hover {
                border: 2px solid #3498db;
            }
            
            QSpinBox {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                font-size: 11pt;
                background-color: white;
                color: #2c3e50;
                min-width: 60px;
            }
            
            QSpinBox:focus {
                border-color: #3498db;
                background-color: #ecf0f1;
            }
            
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                width: 20px;
            }
            
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #3498db;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #d5dbdb);
                border-radius: 5px;
            }
            
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: 2px solid #2980b9;
                width: 22px;
                margin: -6px 0;
                border-radius: 11px;
            }
            
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
                border: 2px solid #3498db;
            }
            
            QCheckBox {
                font-size: 11pt;
                padding: 6px;
                color: #2c3e50;
                background-color: transparent;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border: 2px solid #2980b9;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMCAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTguNSAyLjVMNC4xNjY2NyA2LjgzMzMzTDEuNSA0LjE2NjY3IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #3498db;
            }
            
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #d5dbdb);
                border-top: 1px solid #bdc3c7;
                font-size: 11pt;
                color: #2c3e50;
                padding: 6px;
                font-weight: 500;
            }
            
            QStatusBar QLabel {
                color: #2c3e50;
                background-color: transparent;
                font-size: 10pt;
                padding: 2px 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #ecf0f1;
                min-height: 24px;
                color: #2c3e50;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:0.5 #5dade2, stop:1 #3498db);
                border-radius: 6px;
                margin: 1px;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 12pt;
                border: 2px solid #d5dbdb;
                border-radius: 10px;
                margin-top: 1.5ex;
                padding-top: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 white, stop:1 #f8f9fa);
                color: #2c3e50;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                background-color: white;
                color: #2c3e50;
                border: 1px solid #d5dbdb;
                border-radius: 5px;
            }
            
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bdc3c7, stop:1 #95a5a6);
            }
            
            QSplitter::handle:horizontal {
                width: 4px;
                border-radius: 2px;
            }
            
            QSplitter::handle:vertical {
                height: 4px;
                border-radius: 2px;
            }
            
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
            }
            
            QMenuBar {
                background-color: #34495e;
                color: white;
                font-size: 11pt;
                font-weight: 500;
                border-bottom: 1px solid #2c3e50;
            }
            
            QMenuBar::item {
                spacing: 3px;
                padding: 8px 12px;
                background: transparent;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background-color: #3498db;
            }
            
            QMenu {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                color: #2c3e50;
                font-size: 11pt;
            }
            
            QMenu::item {
                padding: 8px 20px 8px 30px;
                border-bottom: 1px solid #ecf0f1;
            }
            
            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
            
            /* 为图像标签添加样式 */
            ImageLabel {
                border: 3px solid #3498db;
                border-radius: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fafafa, stop:1 #ecf0f1);
            }
        """)

    def zoom_in(self):
        if self.original_image is not None:
            self.image_label.zoom_factor *= 1.2
            self.image_label.zoom_factor = min(5.0, self.image_label.zoom_factor)
            self.update_display()

    def zoom_out(self):
        if self.original_image is not None:
            self.image_label.zoom_factor /= 1.2
            self.image_label.zoom_factor = max(0.1, self.image_label.zoom_factor)
            self.update_display()

    def fit_to_window(self):
        if self.original_image is not None:
            self.image_label.zoom_factor = 1.0
            self.image_label.pan_offset = QPoint(0, 0)
            self.update_display()

    def manual_run_grabcut(self):
        if self.original_image is None:
            self.show_status_message("请先加载图像 📸", 3000)
            return

        if self.current_image_roi_qrect:
            self.run_grabcut(init_with_rect=True)
        elif self.image_label.scribbles_fg or self.image_label.scribbles_bg:
            self.run_grabcut(init_with_mask=True)
        else:
            self.show_status_message("请先框选ROI或添加涂鸦标记 🎯", 3000)

    def clear_scribbles(self):
        if not self.image_label.scribbles_fg and not self.image_label.scribbles_bg:
            self.show_status_message("没有涂鸦需要清除", 2000)
            return

        self.save_state_for_undo()
        self.image_label.scribbles_fg.clear()
        self.image_label.scribbles_bg.clear()
        self.image_label.update()
        self.update_ui_state()
        self.show_status_message("涂鸦已清除 🧹", 2000)

    def update_ui_state(self):
        has_image = self.original_image is not None
        has_result = self.grabcut_mask is not None and np.any(self.grabcut_mask > 0)
        has_scribbles = bool(
            self.image_label.scribbles_fg or self.image_label.scribbles_bg
        )

        self.control_panel.run_grabcut_button.setEnabled(has_image)
        self.control_panel.border_matting_button.setEnabled(has_result)
        self.control_panel.undo_button.setEnabled(len(self.undo_stack) > 1)
        self.control_panel.clear_scribbles_button.setEnabled(has_scribbles)

    def show_status_message(self, message, timeout=0):
        self.status_bar.showMessage(message)
        if timeout > 0:
            self.status_timer.start(timeout)

    def clear_status_message(self):
        self.status_timer.stop()
        self.status_bar.clearMessage()

    def set_interaction_mode(self, mode):
        self.interaction_mode = mode
        cursor_size = self.brush_size + 4
        if mode == "roi":
            self.image_label.setCursor(Qt.CrossCursor)
            self.show_status_message("模式: 框选ROI 📐 拖拽鼠标选择主要目标区域")
        else:
            color = self.fg_color if mode == "fg_scribble" else self.bg_color
            cursor_pixmap = QPixmap(cursor_size, cursor_size)
            cursor_pixmap.fill(Qt.transparent)
            painter = QPainter(cursor_pixmap)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 100))
            painter.drawEllipse(2, 2, self.brush_size, self.brush_size)
            painter.end()
            self.image_label.setCursor(QCursor(cursor_pixmap))
            mode_text = "前景" if mode == "fg_scribble" else "背景"
            icon = "🖌️" if mode == "fg_scribble" else "🗑️"
            self.show_status_message(
                f"模式: {mode_text}涂鸦 {icon} 在图像上标记{mode_text}区域"
            )

    def set_brush_size(self, value):
        self.brush_size = value
        if self.interaction_mode in ["fg_scribble", "bg_scribble"]:
            self.set_interaction_mode(self.interaction_mode)

    def save_state_for_undo(self):
        if self.original_image is None:
            return

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

    def undo_action(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop()
            previous_state = self.undo_stack[-1]

            self.display_image_cv = (
                previous_state["display_image_cv"].copy()
                if previous_state["display_image_cv"] is not None
                else None
            )
            self.grabcut_mask = (
                previous_state["grabcut_mask"].copy()
                if previous_state["grabcut_mask"] is not None
                else None
            )
            self.current_image_roi_qrect = (
                QRect(previous_state["current_image_roi_qrect"])
                if previous_state["current_image_roi_qrect"]
                else None
            )

            self.image_label.scribbles_fg = [
                list(path) for path in previous_state["scribbles_fg"]
            ]
            self.image_label.scribbles_bg = [
                list(path) for path in previous_state["scribbles_bg"]
            ]

            self.update_display()
            self.show_status_message("操作已撤销 ↶", 2000)
        else:
            self.show_status_message("没有更多操作可撤销", 2000)
        self.update_ui_state()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            try:
                img_bytes = np.fromfile(file_path, dtype=np.uint8)
                self.original_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                self.show_status_message(f"错误: 无法读取文件 {file_path}: {e}", 5000)
                self.original_image = None

            if self.original_image is None:
                self.show_status_message(f"错误：无法加载图像 ❌ 文件可能损坏", 5000)
                return

            self.undo_stack = []
            self.display_image_cv = self.original_image.copy()
            self.reset_segmentation_state()
            self.save_state_for_undo()

            self.update_display()
            self.show_status_message(f"图像已加载 ✅ 请框选ROI或开始涂鸦", 3000)
            self.set_interaction_mode("roi")
            self.control_panel.roi_mode_button.setChecked(True)
            self.update_ui_state()

    def reset_segmentation_state(self):
        if self.original_image is not None:
            self.grabcut_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        else:
            self.grabcut_mask = None
        self.current_image_roi_qrect = None
        self.image_label.clear_interactions()

    def update_display(self):
        if self.display_image_cv is None:
            self.image_label.clear()
            self.display_pixmap = None
            return

        img_to_show = self.display_image_cv

        # 应用透明度设置
        if img_to_show.shape[2] == 4:  # BGRA格式
            opacity = self.control_panel.opacity_slider.value() / 100.0
            img_to_show = img_to_show.copy()
            img_to_show[:, :, 3] = (img_to_show[:, :, 3] * opacity).astype(np.uint8)

        # 转换为QImage格式
        if img_to_show.ndim == 2:
            height, width = img_to_show.shape
            bytes_per_line = width
            q_format = QImage.Format_Grayscale8
        elif img_to_show.shape[2] == 3:
            height, width, _ = img_to_show.shape
            bytes_per_line = 3 * width
            q_format = QImage.Format_RGB888
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
        elif img_to_show.shape[2] == 4:
            height, width, _ = img_to_show.shape
            bytes_per_line = 4 * width
            q_format = QImage.Format_RGBA8888
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGRA2RGBA)
        else:
            self.image_label.clear()
            self.display_pixmap = None
            self.show_status_message("错误: 不支持的图像格式", 3000)
            return

        q_image = QImage(img_to_show.data, width, height, bytes_per_line, q_format)

        # 应用缩放
        scaled_size = QSize(
            int(self.image_label.width() * self.image_label.zoom_factor),
            int(self.image_label.height() * self.image_label.zoom_factor),
        )

        self.display_pixmap = QPixmap.fromImage(q_image).scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.image_label.setPixmap(self.display_pixmap)
        self.image_label.update()

    def run_grabcut(self, init_with_rect=False, init_with_mask=False):
        if self.processing_thread and self.processing_thread.isRunning():
            self.show_status_message("处理中，请稍候... ⏳", 1000)
            return
        self.control_panel.progress_bar.setVisible(True)
        self.control_panel.progress_bar.setRange(0, 0)

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

        if init_with_rect and roi_tuple_for_logic:
            mode = "INIT_WITH_RECT"
        elif init_with_mask or fg_scribbles_points or bg_scribbles_points:
            mode = "INIT_WITH_MASK"
        else:
            mode = "INIT_WITH_RECT"

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

    def on_grabcut_finished(
        self, updated_mask, output_binary, status_msg, execution_time
    ):
        self.control_panel.progress_bar.setVisible(False)
        if execution_time >= 0:
            print(f"--- GrabCut 算法执行耗时: {execution_time:.4f} 秒 ---")
        if updated_mask is not None and output_binary is not None:
            self.grabcut_mask = updated_mask

            if self.control_panel.show_mask_overlay.isChecked():
                alpha = (output_binary * 255).astype(np.uint8)
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

    def apply_border_matting(self):
        if self.original_image is None or self.grabcut_mask is None:
            self.show_status_message("请先运行GrabCut得到初步分割结果 🎯", 3000)
            return
        if not np.any(
            (self.grabcut_mask == cv2.GC_FGD) | (self.grabcut_mask == cv2.GC_PR_FGD)
        ):
            self.show_status_message("GrabCut掩码未包含前景，请先运行分割 ❌", 3000)
            return

        self.save_state_for_undo()

        self.control_panel.progress_bar.setVisible(True)
        self.control_panel.progress_bar.setRange(0, 0)
        self.show_status_message("正在应用边缘优化... ✨")
        QApplication.processEvents()

        try:
            matted_image_bgra, status_msg = apply_border_matting_logic(
                self.original_image, self.grabcut_mask
            )

            if matted_image_bgra is not None:
                self.display_image_cv = matted_image_bgra
                self.update_display()
                self.show_status_message("边缘优化完成 ✨", 3000)
            else:
                output_mask_binary = np.where(
                    (self.grabcut_mask == cv2.GC_BGD)
                    | (self.grabcut_mask == cv2.GC_PR_BGD),
                    0,
                    1,
                ).astype("uint8")
                self.display_image_cv = (
                    self.original_image * output_mask_binary[:, :, np.newaxis]
                )
                self.update_display()
                self.show_status_message(status_msg, 3000)
        except Exception as e:
            self.show_status_message(f"边缘优化失败: {str(e)}", 3000)
        finally:
            self.control_panel.progress_bar.setVisible(False)

    def save_result(self):
        if self.display_image_cv is None:
            self.show_status_message("没有可保存的结果", 2000)
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG文件 (*.png);;JPEG文件 (*.jpg);;BMP文件 (*.bmp)"
        )

        if file_path:
            try:
                # 转换颜色格式用于保存
                if self.display_image_cv.shape[2] == 4:
                    save_image = cv2.cvtColor(self.display_image_cv, cv2.COLOR_BGRA2BGR)
                else:
                    save_image = self.display_image_cv

                cv2.imwrite(file_path, save_image)
                self.show_status_message(f"结果已保存 💾 {file_path}", 3000)
            except Exception as e:
                self.show_status_message(f"保存失败: {str(e)}", 3000)

    def reset_current_image_state(self):
        if self.original_image is None:
            self.show_status_message("没有加载图像可重置", 2000)
            return

        self.save_state_for_undo()
        self.display_image_cv = self.original_image.copy()
        self.reset_segmentation_state()
        self.update_display()
        self.control_panel.roi_mode_button.setChecked(True)
        self.show_status_message("当前图像状态已重置 🔄", 2000)

    def reset_all(self):
        self.original_image = None
        self.display_image_cv = None
        self.display_pixmap = None
        self.reset_segmentation_state()
        self.image_label.clear()
        self.undo_stack = []
        self.control_panel.roi_mode_button.setChecked(True)
        self.update_ui_state()
        self.show_status_message("已重置所有，请加载新图像 🗑️", 3000)
        self.coord_label.setText("")

    def update_status_bar_coords(self, img_coord_qpoint):
        if img_coord_qpoint:
            self.coord_label.setText(
                f"坐标: ({img_coord_qpoint.x()}, {img_coord_qpoint.y()})"
            )
        else:
            self.coord_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.quit()
            self.processing_thread.wait()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("交互式图像分割系统")
    app.setApplicationVersion("2.0")

    main_window = InteractiveSegmentationApp()
    main_window.show()

    sys.exit(app.exec_())
