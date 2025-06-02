import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSizePolicy, QAction, QMenuBar, QStatusBar, QSpinBox, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QCursor
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from segmentation import run_grabcut_logic, apply_border_matting_logic


class ImageLabel(QLabel):
    """自定义QLabel用于显示图像并处理鼠标交互事件"""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app  # 主窗口的引用
        self.setMinimumSize(400, 300)  # 设置最小尺寸
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)  # 持续追踪鼠标事件，即使没有按下按钮

        self.current_roi_rect_label = QRect()  # ROI in label coordinates, for drawing
        self.drawing_roi = False
        # [[QPoint, QPoint,...], ...] 存储前景涂鸦线段 (image coords)
        self.scribbles_fg = []
        # [[QPoint, QPoint,...], ...] 存储背景涂鸦线段 (image coords)
        self.scribbles_bg = []
        self.current_scribble_img_coords = []  # Current scribble in image coordinates

        self.setCursor(Qt.CrossCursor)  # 默认十字光标

    def mousePressEvent(self, event):
        if self.parent_app.original_image is None:
            return

        pos_in_label = event.pos()  # Mouse position in Label coordinates
        img_coord = self.map_label_to_image_coords(
            pos_in_label)  # Convert to image coordinates

        if not img_coord:  # Click position not within the valid image area on the label
            return

        if self.parent_app.interaction_mode == "roi":
            if event.button() == Qt.LeftButton:
                # 在开始新的ROI绘制前保存状态，以便可以撤销这个ROI操作
                self.parent_app.save_state_for_undo()
                self.drawing_roi = True
                self.current_roi_rect_label.setTopLeft(pos_in_label)
                self.current_roi_rect_label.setBottomRight(pos_in_label)
        elif self.parent_app.interaction_mode in ["fg_scribble", "bg_scribble"]:
            if event.button() == Qt.LeftButton:
                # 在开始新的涂鸦前保存状态
                self.parent_app.save_state_for_undo()
                # Start new scribble with image coordinates
                self.current_scribble_img_coords = [img_coord]

        self.update()

    def mouseMoveEvent(self, event):
        if self.parent_app.original_image is None:
            return

        pos_in_label = event.pos()
        img_coord = self.map_label_to_image_coords(pos_in_label)
        self.parent_app.update_status_bar_coords(img_coord)

        if self.parent_app.interaction_mode == "roi":
            if self.drawing_roi:
                # ROI is drawn in label coordinates directly
                self.current_roi_rect_label.setBottomRight(pos_in_label)
        elif self.parent_app.interaction_mode in ["fg_scribble", "bg_scribble"]:
            if event.buttons() & Qt.LeftButton and self.current_scribble_img_coords:
                if img_coord:  # Ensure coordinate is valid
                    self.current_scribble_img_coords.append(img_coord)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.parent_app.original_image is None:
            return

        if self.parent_app.interaction_mode == "roi":
            if event.button() == Qt.LeftButton and self.drawing_roi:
                self.drawing_roi = False
                normalized_label_roi = self.current_roi_rect_label.normalized()

                top_left_img = self.map_label_to_image_coords(
                    normalized_label_roi.topLeft())
                bottom_right_img = self.map_label_to_image_coords(
                    normalized_label_roi.bottomRight())

                if top_left_img and bottom_right_img:
                    self.parent_app.current_image_roi_qrect = QRect(
                        top_left_img, bottom_right_img).normalized()
                    if self.parent_app.current_image_roi_qrect.width() > 0 and self.parent_app.current_image_roi_qrect.height() > 0:
                        self.parent_app.run_grabcut(init_with_rect=True)
                    else:
                        self.parent_app.current_image_roi_qrect = None
                        self.parent_app.status_bar.showMessage(
                            "ROI选择过小或无效。")
                else:
                    self.parent_app.current_image_roi_qrect = None
                self.current_roi_rect_label = QRect()

        elif self.parent_app.interaction_mode in ["fg_scribble", "bg_scribble"]:
            if event.button() == Qt.LeftButton and self.current_scribble_img_coords:
                if len(self.current_scribble_img_coords) > 1:
                    if self.parent_app.interaction_mode == "fg_scribble":
                        self.scribbles_fg.append(
                            list(self.current_scribble_img_coords))
                    else:
                        self.scribbles_bg.append(
                            list(self.current_scribble_img_coords))
                elif len(self.current_scribble_img_coords) == 1:
                    point = self.current_scribble_img_coords[0]
                    tiny_segment = [point, QPoint(
                        point.x() + 1, point.y() + 1)]
                    if self.parent_app.interaction_mode == "fg_scribble":
                        self.scribbles_fg.append(tiny_segment)
                    else:
                        self.scribbles_bg.append(tiny_segment)

                self.current_scribble_img_coords = []
                if self.parent_app.grabcut_mask is not None:  # 确保GrabCut已运行或至少有初始掩码
                    # 涂鸦后自动运行GrabCut进行优化
                    self.parent_app.run_grabcut(init_with_mask=True)
                elif self.parent_app.original_image is not None:  # 如果是第一次涂鸦且没有ROI
                    # 这种情况可能需要用户先定义ROI，或者我们允许直接用涂鸦初始化
                    # 目前，我们假设如果grabcut_mask为None，但有图像，则需要先有ROI
                    # 如果要允许无ROI直接涂鸦，run_grabcut逻辑需要调整
                    self.parent_app.status_bar.showMessage(
                        "建议先框选ROI，或确保GrabCut已基于ROI初始化。")

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.parent_app.display_pixmap is None or self.parent_app.display_pixmap.isNull():
            return

        painter = QPainter(self)

        if self.drawing_roi and not self.current_roi_rect_label.isNull():
            pen = QPen(QColor(0, 255, 255, 180), 2,
                       Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_roi_rect_label.normalized())

        pen_fg = QPen(self.parent_app.fg_color, self.parent_app.brush_size,
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_fg)
        for scribble_path_img_coords in self.scribbles_fg:
            if len(scribble_path_img_coords) > 1:
                poly_points_label = [self.map_image_to_label_coords(
                    pt) for pt in scribble_path_img_coords]
                poly_points_label = [
                    pt for pt in poly_points_label if pt is not None]
                if len(poly_points_label) > 1:
                    painter.drawPolyline(*poly_points_label)
            elif len(scribble_path_img_coords) == 1:
                pt_label = self.map_image_to_label_coords(
                    scribble_path_img_coords[0])
                if pt_label:
                    painter.drawPoint(pt_label)

        pen_bg = QPen(self.parent_app.bg_color, self.parent_app.brush_size,
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_bg)
        for scribble_path_img_coords in self.scribbles_bg:
            if len(scribble_path_img_coords) > 1:
                poly_points_label = [self.map_image_to_label_coords(
                    pt) for pt in scribble_path_img_coords]
                poly_points_label = [
                    pt for pt in poly_points_label if pt is not None]
                if len(poly_points_label) > 1:
                    painter.drawPolyline(*poly_points_label)
            elif len(scribble_path_img_coords) == 1:
                pt_label = self.map_image_to_label_coords(
                    scribble_path_img_coords[0])
                if pt_label:
                    painter.drawPoint(pt_label)

        if self.current_scribble_img_coords and len(self.current_scribble_img_coords) > 1:
            current_pen = pen_fg if self.parent_app.interaction_mode == "fg_scribble" else pen_bg
            painter.setPen(current_pen)
            poly_points_label = [self.map_image_to_label_coords(
                pt) for pt in self.current_scribble_img_coords]
            poly_points_label = [
                pt for pt in poly_points_label if pt is not None]
            if len(poly_points_label) > 1:
                painter.drawPolyline(*poly_points_label)

    def map_label_to_image_coords(self, label_pos):
        if self.parent_app.original_image is None or self.parent_app.display_pixmap is None or self.parent_app.display_pixmap.isNull():
            return None

        pixmap_w = self.parent_app.display_pixmap.width()
        pixmap_h = self.parent_app.display_pixmap.height()
        label_w = self.width()
        label_h = self.height()

        offset_x = (label_w - pixmap_w) / 2
        offset_y = (label_h - pixmap_h) / 2

        pixmap_x = label_pos.x() - offset_x
        pixmap_y = label_pos.y() - offset_y

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
        if self.parent_app.original_image is None or self.parent_app.display_pixmap is None or self.parent_app.display_pixmap.isNull():
            return None

        pixmap_w = self.parent_app.display_pixmap.width()
        pixmap_h = self.parent_app.display_pixmap.height()
        label_w = self.width()
        label_h = self.height()

        offset_x = (label_w - pixmap_w) / 2
        offset_y = (label_h - pixmap_h) / 2

        orig_h, orig_w = self.parent_app.original_image.shape[:2]
        if orig_w == 0 or orig_h == 0:
            return None

        scale_x = pixmap_w / orig_w
        scale_y = pixmap_h / orig_h

        pixmap_x_float = image_pos.x() * scale_x
        pixmap_y_float = image_pos.y() * scale_y

        label_x = int(pixmap_x_float + offset_x)
        label_y = int(pixmap_y_float + offset_y)

        return QPoint(label_x, label_y)

    def clear_interactions(self):
        self.current_roi_rect_label = QRect()
        self.scribbles_fg.clear()
        self.scribbles_bg.clear()
        self.current_scribble_img_coords = []
        self.update()


class InteractiveSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互式图像分割系统")
        self.setGeometry(100, 100, 1200, 800)

        self.original_image = None
        self.display_image_cv = None
        self.display_pixmap = None

        self.grabcut_mask = None
        self.current_image_roi_qrect = None

        self.interaction_mode = "roi"
        self.fg_color = QColor(0, 255, 0, 180)
        self.bg_color = QColor(255, 0, 0, 180)
        self.brush_size = 5

        # 用于存储状态以实现撤销功能
        self.undo_stack = []

        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        load_action = QAction(QIcon.fromTheme(
            "document-open"), "加载图像...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        save_action = QAction(QIcon.fromTheme(
            "document-save"), "保存结果...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QAction(QIcon.fromTheme("application-exit"), "退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(200)

        mode_group_box = QWidget()
        mode_layout = QVBoxLayout()
        mode_group_box.setLayout(mode_layout)
        mode_layout.addWidget(QLabel("<b>交互模式:</b>"))
        self.mode_button_group = QButtonGroup(self)
        self.roi_mode_button = QRadioButton("框选ROI")
        self.roi_mode_button.setChecked(True)
        self.roi_mode_button.toggled.connect(
            lambda: self.set_interaction_mode("roi"))
        mode_layout.addWidget(self.roi_mode_button)
        self.mode_button_group.addButton(self.roi_mode_button)
        self.fg_scribble_button = QRadioButton("前景涂鸦")
        self.fg_scribble_button.toggled.connect(
            lambda: self.set_interaction_mode("fg_scribble"))
        mode_layout.addWidget(self.fg_scribble_button)
        self.mode_button_group.addButton(self.fg_scribble_button)
        self.bg_scribble_button = QRadioButton("背景涂鸦")
        self.bg_scribble_button.toggled.connect(
            lambda: self.set_interaction_mode("bg_scribble"))
        mode_layout.addWidget(self.bg_scribble_button)
        self.mode_button_group.addButton(self.bg_scribble_button)
        controls_layout.addWidget(mode_group_box)
        controls_layout.addSpacing(10)

        brush_size_label = QLabel("笔刷大小:")
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 50)
        self.brush_size_spinbox.setValue(self.brush_size)
        self.brush_size_spinbox.valueChanged.connect(self.set_brush_size)
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(brush_size_label)
        brush_layout.addWidget(self.brush_size_spinbox)
        controls_layout.addLayout(brush_layout)
        controls_layout.addSpacing(10)

        # --- 新增和修改的按钮 ---
        self.undo_button = QPushButton("撤销 (Undo)")
        self.undo_button.clicked.connect(self.undo_action)
        controls_layout.addWidget(self.undo_button)

        self.border_matting_button = QPushButton("边缘优化 (Matting)")
        self.border_matting_button.clicked.connect(self.apply_border_matting)
        controls_layout.addWidget(self.border_matting_button)

        self.reset_current_button = QPushButton("重置当前图像")
        self.reset_current_button.clicked.connect(
            self.reset_current_image_state)
        controls_layout.addWidget(self.reset_current_button)

        self.reset_button = QPushButton("重置所有 (清空)")
        self.reset_button.clicked.connect(self.reset_all)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addStretch()

        self.update_undo_button_state()  # 初始化撤销按钮状态

        self.image_label = ImageLabel(self)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout = QHBoxLayout()
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.image_label, 1)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("请先加载图像。")
        self.coord_label = QLabel("")
        self.status_bar.addPermanentWidget(self.coord_label)
        self.set_interaction_mode("roi")

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; } QLabel { font-size: 10pt; }
            QPushButton { background-color: #4CAF50; border: none; color: white; padding: 8px 16px;
                          text-align: center; font-size: 10pt; margin: 4px 2px; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; } QPushButton:pressed { background-color: #3e8e41; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
            QRadioButton { font-size: 10pt; margin-bottom: 5px; }
            QSpinBox { padding: 4px; border: 1px solid #ccc; border-radius: 3px; font-size: 10pt; }
        """)

    def set_interaction_mode(self, mode):
        self.interaction_mode = mode
        cursor_size = self.brush_size + 4
        if mode == "roi":
            self.image_label.setCursor(Qt.CrossCursor)
            self.status_bar.showMessage("模式: 框选ROI。拖拽鼠标选择主要目标区域。")
        else:
            color = self.fg_color if mode == "fg_scribble" else self.bg_color
            cursor_pixmap = QPixmap(cursor_size, cursor_size)
            cursor_pixmap.fill(Qt.transparent)
            painter = QPainter(cursor_pixmap)
            painter.setPen(QPen(color, 2))
            painter.setBrush(
                QColor(color.red(), color.green(), color.blue(), 100))
            painter.drawEllipse(2, 2, self.brush_size, self.brush_size)
            painter.end()
            self.image_label.setCursor(QCursor(cursor_pixmap))
            mode_text = "前景" if mode == "fg_scribble" else "背景"
            self.status_bar.showMessage(
                f"模式: {mode_text}涂鸦。在图像上标记明确的{mode_text}区域。")

    def set_brush_size(self, value):
        self.brush_size = value
        if self.interaction_mode in ["fg_scribble", "bg_scribble"]:
            self.set_interaction_mode(self.interaction_mode)

    def save_state_for_undo(self):
        """保存当前状态到撤销栈"""
        if self.original_image is None:
            return

        state = {
            'display_image_cv': self.display_image_cv.copy() if self.display_image_cv is not None else None,
            'grabcut_mask': self.grabcut_mask.copy() if self.grabcut_mask is not None else None,
            'current_image_roi_qrect': self.current_image_roi_qrect.normalized() if self.current_image_roi_qrect else None,
            'scribbles_fg': [list(path) for path in self.image_label.scribbles_fg],
            'scribbles_bg': [list(path) for path in self.image_label.scribbles_bg],
        }
        self.undo_stack.append(state)
        self.update_undo_button_state()

    def undo_action(self):
        """撤销上一步操作"""
        if len(self.undo_stack) > 1:  # 至少需要两个状态才能撤销 (当前状态和前一个状态)
            self.undo_stack.pop()  # 移除当前状态
            previous_state = self.undo_stack[-1]  # 获取栈顶的状态作为新的当前状态

            self.display_image_cv = previous_state['display_image_cv'].copy(
            ) if previous_state['display_image_cv'] is not None else None
            self.grabcut_mask = previous_state['grabcut_mask'].copy(
            ) if previous_state['grabcut_mask'] is not None else None
            self.current_image_roi_qrect = QRect(
                previous_state['current_image_roi_qrect']) if previous_state['current_image_roi_qrect'] else None

            self.image_label.scribbles_fg = [
                list(path) for path in previous_state['scribbles_fg']]
            self.image_label.scribbles_bg = [
                list(path) for path in previous_state['scribbles_bg']]

            self.update_display()
            self.status_bar.showMessage("操作已撤销。")
        else:
            self.status_bar.showMessage("没有更多操作可撤销。")
        self.update_undo_button_state()

    def update_undo_button_state(self):
        """根据撤销栈状态更新撤销按钮的可用性"""
        if hasattr(self, 'undo_button'):
            self.undo_button.setEnabled(len(self.undo_stack) > 1)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                img_bytes = np.fromfile(file_path, dtype=np.uint8)
                self.original_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                self.status_bar.showMessage(f"错误: 无法读取文件 {file_path}: {e}")
                self.original_image = None

            if self.original_image is None:
                self.status_bar.showMessage(
                    f"错误：无法加载图像: {file_path}。文件可能损坏或路径不正确。")
                return

            self.undo_stack = []
            self.display_image_cv = self.original_image.copy()
            self.reset_segmentation_state()
            self.save_state_for_undo()

            self.update_display()
            self.status_bar.showMessage(f"图像已加载: {file_path}。请先框选ROI或开始涂鸦。")
            self.set_interaction_mode("roi")
            self.roi_mode_button.setChecked(True)
            self.update_undo_button_state()

    def reset_segmentation_state(self):
        if self.original_image is not None:
            self.grabcut_mask = np.zeros(
                self.original_image.shape[:2], dtype=np.uint8)
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
        q_format = None
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
            img_to_show = cv2.cvtColor(
                img_to_show, cv2.COLOR_BGRA2RGBA)
        else:
            self.image_label.clear()
            self.display_pixmap = None
            self.status_bar.showMessage("错误: 不支持的图像格式进行显示。")
            return

        q_image = QImage(img_to_show.data, width, height,
                         bytes_per_line, q_format)

        self.display_pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(self.display_pixmap)
        self.image_label.update()

    def run_grabcut(self, init_with_rect=False, init_with_mask=False):
        if self.original_image is None:
            self.status_bar.showMessage("请先加载图像。")
            return
        if self.grabcut_mask is None:
            self.grabcut_mask = np.zeros(
                self.original_image.shape[:2], dtype=np.uint8)

        # 在执行GrabCut之前保存状态，如果不是因为撤销而运行

        roi_tuple_for_logic = None
        if self.current_image_roi_qrect:
            r = self.current_image_roi_qrect
            roi_tuple_for_logic = (r.x(), r.y(), r.width(), r.height())

        fg_scribbles_points = [[(p.x(), p.y()) for p in path]
                               for path in self.image_label.scribbles_fg]
        bg_scribbles_points = [[(p.x(), p.y()) for p in path]
                               for path in self.image_label.scribbles_bg]

        current_mode_option = ""
        if init_with_rect and roi_tuple_for_logic and roi_tuple_for_logic[2] > 0 and roi_tuple_for_logic[3] > 0:
            current_mode_option = "INIT_WITH_RECT"
            self.status_bar.showMessage("使用ROI初始化GrabCut...")
        elif init_with_mask:
            is_mask_initialized_by_roi = np.any(
                (self.grabcut_mask == cv2.GC_PR_FGD) | (self.grabcut_mask == cv2.GC_PR_BGD))
            if not fg_scribbles_points and not bg_scribbles_points and not is_mask_initialized_by_roi:
                if not self.current_image_roi_qrect:
                    self.status_bar.showMessage("请先提供ROI或涂鸦标记以优化。")
                    return
            current_mode_option = "INIT_WITH_MASK"
            self.status_bar.showMessage("使用涂鸦/当前掩码优化GrabCut...")
        else:
            if fg_scribbles_points or bg_scribbles_points or np.any(self.grabcut_mask != 0):
                current_mode_option = "INIT_WITH_MASK"
                self.status_bar.showMessage("优化GrabCut...")
            elif roi_tuple_for_logic and roi_tuple_for_logic[2] > 0 and roi_tuple_for_logic[3] > 0:
                current_mode_option = "INIT_WITH_RECT"
                self.status_bar.showMessage("使用ROI初始化GrabCut...")
            else:
                self.status_bar.showMessage("请先进行ROI选择或涂鸦。")
                return

        updated_mask, output_binary, status_msg = run_grabcut_logic(
            self.original_image, self.grabcut_mask, roi_tuple_for_logic,
            fg_scribbles_points, bg_scribbles_points, self.brush_size,
            current_mode_option
        )

        if updated_mask is not None and output_binary is not None:
            self.grabcut_mask = updated_mask
            self.display_image_cv = self.original_image * \
                output_binary[:, :, np.newaxis]
            self.update_display()
            self.status_bar.showMessage(status_msg)
            # GrabCut成功运行后，保存新状态
            self.save_state_for_undo()
        else:
            self.status_bar.showMessage(status_msg)

    def apply_border_matting(self):
        if self.original_image is None or self.grabcut_mask is None:
            self.status_bar.showMessage("请先运行GrabCut得到初步分割结果。")
            return
        if not np.any((self.grabcut_mask == cv2.GC_FGD) | (self.grabcut_mask == cv2.GC_PR_FGD)):
            self.status_bar.showMessage("GrabCut掩码未包含明确前景。请先运行或优化GrabCut。")
            return

        self.save_state_for_undo()  # 在应用抠图前保存状态

        self.status_bar.showMessage("正在应用边缘优化 (Border Matting)...")
        QApplication.processEvents()

        matted_image_bgra, status_msg = apply_border_matting_logic(
            self.original_image, self.grabcut_mask
        )

        if matted_image_bgra is not None:
            self.display_image_cv = matted_image_bgra
            self.update_display()
            self.status_bar.showMessage(status_msg)
            # 边缘优化成功后，保存新状态 (这一步已经在save_state_for_undo()中完成，如果操作成功，它就是新状态)
            # 但如果希望抠图操作本身也能被撤销，那么需要在成功后再次保存。
            # 当前设计是在操作前保存，如果操作修改了状态，则修改后的状态成为栈顶。
            # 如果抠图是一个独立步骤，并且想要撤销到抠图前，则抠图操作本身不应该再调用save_state_for_undo
            # 而是在调用apply_border_matting之前已经保存了状态。
            # 检查：save_state_for_undo() 已经在函数开头调用，是正确的。
        else:
            self.status_bar.showMessage(status_msg)
            # 如果抠图失败，我们可能希望撤销栈回到抠图尝试之前的状态
            # 目前，由于在开始时保存了状态，如果抠图失败，最新的状态（栈顶）就是抠图前的状态。
            # 可以选择pop掉这个“尝试抠图但失败”的状态，或者保留它。
            # 如果保留，用户可以撤销到抠图前的状态。
            # 如果pop，则相当于抠图操作“未发生”。
            # 为了简化，我们目前保留它，用户可以手动撤销。
            output_mask_binary = np.where((self.grabcut_mask == cv2.GC_BGD) | (
                self.grabcut_mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
            self.display_image_cv = self.original_image * \
                output_mask_binary[:, :, np.newaxis]
            self.update_display()

    def reset_current_image_state(self):
        """重置当前图像到其加载时的初始状态，清除所有修改。"""
        if self.original_image is None:
            self.status_bar.showMessage("没有加载图像可重置。")
            return

        self.save_state_for_undo()  # 保存当前状态，以便可以撤销“重置”操作

        self.display_image_cv = self.original_image.copy()
        self.reset_segmentation_state()  # 清除掩码、ROI、涂鸦

        self.update_display()
        self.roi_mode_button.setChecked(True)  # 重置交互模式
        # set_interaction_mode 会更新状态栏消息
        self.status_bar.showMessage("当前图像状态已重置。")
        # self.update_undo_button_state() # save_state_for_undo 内部会调用

    def reset_all(self):
        """重置所有，包括卸载图像。"""
        self.save_state_for_undo()  # 允许撤销“重置所有”回到之前的状态

        self.original_image = None
        self.display_image_cv = None
        self.display_pixmap = None
        self.reset_segmentation_state()
        self.image_label.clear()

        self.undo_stack = []  # 清空撤销栈，因为图像已卸载
        self.save_state_for_undo()  # 对于“无图像”状态，也推入一个空状态，确保undo_stack不为空
        # 或者，在load_image时处理undo_stack的初始化
        # 当前load_image会清空并加入初始状态，所以这里清空即可

        self.roi_mode_button.setChecked(True)
        self.status_bar.showMessage("已重置所有。请加载新图像。")
        self.coord_label.setText("")
        self.update_undo_button_state()

    def save_result(self):
        if self.display_image_cv is None:
            self.status_bar.showMessage("没有可保存的结果。")
            return

        image_to_save = self.display_image_cv
        if image_to_save.ndim == 3 or image_to_save.shape[2] == 3:
            if self.grabcut_mask is not None:
                alpha = np.where((self.grabcut_mask == cv2.GC_FGD) | (self.grabcut_mask == cv2.GC_PR_FGD),
                                 255, 0).astype(np.uint8)
                b, g, r = cv2.split(self.original_image)  # 使用原始图像颜色
                image_to_save = cv2.merge((b, g, r, alpha))
            else:
                self.status_bar.showMessage("无法确定Alpha通道，结果将不含透明度。")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg)")
        if file_path:
            try:
                if file_path.lower().endswith('.png') and image_to_save.shape[2] == 3:
                    self.status_bar.showMessage("保存为PNG但Alpha通道未生成，将无透明度。")
                cv2.imwrite(file_path, image_to_save)
                self.status_bar.showMessage(f"结果已保存到: {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"保存失败: {e}")

    def update_status_bar_coords(self, img_coord_qpoint):
        if img_coord_qpoint:
            self.coord_label.setText(
                f"图像坐标: ({img_coord_qpoint.x()}, {img_coord_qpoint.y()})")
        else:
            self.coord_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = InteractiveSegmentationApp()
    main_window.show()
    sys.exit(app.exec_())
