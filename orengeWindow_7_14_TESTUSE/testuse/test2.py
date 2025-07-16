import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                            QSlider, QComboBox, QColorDialog, QGroupBox, 
                            QListWidget, QListWidgetItem, QGraphicsOpacityEffect)
from PyQt5.QtCore import (Qt, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, 
                         QSequentialAnimationGroup, QRectF, QPoint)
from PyQt5.QtGui import (QColor, QLinearGradient, QPainter, QPainterPath, 
                        QBrush, QPen, QFont, QPixmap)
from PyQt5.QtCore import pyqtProperty

class AnimatedButton(QPushButton):
    """带有悬停动画的按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        
        # 设置初始样式
        self.normal_style = """
            QPushButton {
                background-color: rgba(52, 152, 219, 150);
                color: white;
                border: none;
                border-radius: 15px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(41, 128, 185, 200);
            }
        """
        self.setStyleSheet(self.normal_style)
        
        # 创建动画效果
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutQuad)
        
        # 创建透明度效果
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.opacity_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.opacity_animation.setDuration(300)
        
    def enterEvent(self, event):
        """鼠标进入时动画"""
        self.animation.stop()
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(self.geometry().adjusted(-5, -5, 5, 5))
        self.animation.start()
        
        self.opacity_animation.stop()
        self.opacity_animation.setStartValue(0.8)
        self.opacity_animation.setEndValue(1.0)
        self.opacity_animation.start()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开时动画"""
        self.animation.stop()
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(self.geometry().adjusted(5, 5, -5, -5))
        self.animation.start()
        
        self.opacity_animation.stop()
        self.opacity_animation.setStartValue(1.0)
        self.opacity_animation.setEndValue(0.8)
        self.opacity_animation.start()
        super().leaveEvent(event)

class RoundedWidget(QWidget):
    """带有圆角和阴影效果的Widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.border_radius = 15
        self.shadow_offset = 5
        self.shadow_color = QColor(0, 0, 0, 80)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 转换为QRectF
        rect = QRectF(self.rect())
        
        # 绘制阴影
        path = QPainterPath()
        path.addRoundedRect(
            rect.adjusted(
                self.shadow_offset, self.shadow_offset,
                -self.shadow_offset, -self.shadow_offset
            ),
            self.border_radius, self.border_radius
        )
        
        painter.fillPath(path, self.shadow_color)
        
        # 绘制背景
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 230))
        gradient.setColorAt(1, QColor(240, 240, 240, 230))
        
        bg_path = QPainterPath()
        bg_path.addRoundedRect(
            rect.adjusted(0, 0, -self.shadow_offset, -self.shadow_offset),
            self.border_radius, self.border_radius
        )
        
        painter.fillPath(bg_path, QBrush(gradient))
        
        # 绘制边框
        pen = QPen(QColor(200, 200, 200, 150), 1)
        painter.setPen(pen)
        painter.drawPath(bg_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("现代化UI应用")
        self.setGeometry(100, 100, 900, 600)
        
        # 设置窗口背景为渐变
        self.setAutoFillBackground(True)
        p = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(52, 152, 219))
        gradient.setColorAt(1, QColor(155, 89, 182))
        p.setBrush(self.backgroundRole(), QBrush(gradient))
        self.setPalette(p)
        
        # 创建中心部件和主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # 创建侧边栏
        self.create_sidebar()
        
        # 创建堆叠窗口部件(用于管理多个界面)
        self.stacked_widget = QStackedWidget()
        
        # 添加不同的界面
        self.main_page = MainPage()
        self.settings_page = SettingsPage()
        self.about_page = AboutPage()
        
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.about_page)
        
        # 将堆叠窗口添加到主布局
        self.main_layout.addWidget(self.stacked_widget, 4)
        
        # 页面切换动画
        self.current_index = 0
        self.animation_group = QParallelAnimationGroup()
        
    def create_sidebar(self):
        """创建美化后的侧边栏菜单"""
        self.sidebar = RoundedWidget()
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(15)
        
        # 标题
        title = QLabel("导航菜单")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 18px;
                font-weight: bold;
                padding: 10px 0;
            }
        """)
        
        # 创建菜单列表
        self.menu_list = QListWidget()
        self.menu_list.setFrameShape(QListWidget.NoFrame)
        self.menu_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
            }
            QListWidget::item {
                color: #34495e;
                padding: 12px 10px;
                border-radius: 10px;
                margin: 3px 0;
            }
            QListWidget::item:hover {
                background-color: rgba(52, 152, 219, 100);
                color: white;
            }
            QListWidget::item:selected {
                background-color: rgba(52, 152, 219, 150);
                color: white;
            }
        """)
        
        # 添加菜单项
        menu_items = ["主页", "设置", "关于"]
        for item_text in menu_items:
            item = QListWidgetItem(item_text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(QFont("Arial", 12))
            self.menu_list.addItem(item)
        
        # 连接列表项点击事件
        self.menu_list.currentRowChanged.connect(self.change_page_with_animation)
        
        # 将菜单添加到侧边栏布局
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(self.menu_list)
        sidebar_layout.addStretch()
        
        # 将侧边栏添加到主布局
        self.main_layout.addWidget(self.sidebar, 1)
    
    def change_page_with_animation(self, index):
        """带有动画效果的页面切换"""
        if index == self.current_index:
            return
            
        # 创建动画组
        self.animation_group = QParallelAnimationGroup()
        
        # 当前页面淡出
        fade_out = QPropertyAnimation(self.stacked_widget.widget(self.current_index), b"windowOpacity")
        fade_out.setDuration(300)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        
        # 新页面淡入
        fade_in = QPropertyAnimation(self.stacked_widget.widget(index), b"windowOpacity")
        fade_in.setDuration(300)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        
        # 滑动动画
        slide = QPropertyAnimation(self.stacked_widget, b"pos")
        slide.setDuration(400)
        slide.setEasingCurve(QEasingCurve.OutCubic)
        
        current_pos = self.stacked_widget.pos()
        if index > self.current_index:
            slide.setStartValue(current_pos)
            slide.setEndValue(current_pos - QPoint(50, 0))
        else:
            slide.setStartValue(current_pos)
            slide.setEndValue(current_pos + QPoint(50, 0))
        
        # 重置位置动画
        reset_pos = QPropertyAnimation(self.stacked_widget, b"pos")
        reset_pos.setDuration(0)
        reset_pos.setStartValue(slide.endValue())
        reset_pos.setEndValue(current_pos)
        
        # 顺序执行动画
        seq_group = QSequentialAnimationGroup()
        seq_group.addAnimation(fade_out)
        seq_group.addAnimation(slide)
        
        # 并行执行动画
        self.animation_group.addAnimation(seq_group)
        self.animation_group.addAnimation(fade_in)
        self.animation_group.addAnimation(reset_pos)
        
        # 设置新页面
        self.stacked_widget.setCurrentIndex(index)
        self.current_index = index
        
        # 启动动画
        self.animation_group.start()

class MainPage(RoundedWidget):
    """主界面"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 标题
        title = QLabel("欢迎来到主界面")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px 0;
            }
        """)
        
        # 内容
        content = QLabel("""
            <p style="color: #34495e; font-size: 14px; line-height: 1.5;">
                这是一个现代化的PyQt5应用界面，展示了渐变背景、圆角边框、动画效果等UI美化技术。
            </p>
        """)
        content.setAlignment(Qt.AlignCenter)
        content.setWordWrap(True)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.random_button = AnimatedButton("生成随机数")
        self.random_button.clicked.connect(self.generate_random_number)
        
        self.info_button = AnimatedButton("显示信息")
        self.info_button.clicked.connect(self.show_info)
        
        button_layout.addStretch()
        button_layout.addWidget(self.random_button)
        button_layout.addWidget(self.info_button)
        button_layout.addStretch()
        
        # 结果标签
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: rgba(236, 240, 241, 150);
                border-radius: 10px;
                padding: 15px;
                color: #2c3e50;
                font-size: 16px;
            }
        """)
        
        # 添加所有部件到布局
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(content)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_label)
        layout.addStretch()
    
    def generate_random_number(self):
        import random
        number = random.randint(1, 100)
        self.result_label.setText(f"<b>随机数:</b> {number}")
    
    def show_info(self):
        self.result_label.setText("""
            <p style="line-height: 1.5;">
                <b>现代化UI特性:</b><br>
                • 渐变背景和圆角边框<br>
                • 平滑的动画过渡效果<br>
                • 悬停交互反馈<br>
                • 现代化的配色方案
            </p>
        """)

class SettingsPage(RoundedWidget):
    """设置界面"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 标题
        title = QLabel("设置")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px 0;
            }
        """)
        
        # 创建设置内容
        settings_group = QGroupBox("应用设置")
        settings_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(189, 195, 199, 150);
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
                color: #2c3e50;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(15)
        
        # 主题颜色设置
        theme_layout = QHBoxLayout()
        theme_layout.setSpacing(10)
        
        theme_label = QLabel("主题颜色:")
        theme_label.setStyleSheet("color: #34495e;")
        
        self.theme_color_preview = QLabel()
        self.theme_color_preview.setFixedSize(30, 30)
        self.theme_color_preview.setStyleSheet("""
            QLabel {
                background-color: rgb(52, 152, 219);
                border-radius: 15px;
                border: 2px solid white;
            }
        """)
        
        theme_button = AnimatedButton("选择颜色")
        theme_button.clicked.connect(self.select_theme_color)
        
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_color_preview)
        theme_layout.addWidget(theme_button)
        theme_layout.addStretch()
        
        # 音量设置
        volume_layout = QHBoxLayout()
        volume_layout.setSpacing(10)
        
        volume_label = QLabel("音量:")
        volume_label.setStyleSheet("color: #34495e;")
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(75)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: rgba(189, 195, 199, 100);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -4px 0;
                background: rgb(52, 152, 219);
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: rgb(52, 152, 219);
                border-radius: 4px;
            }
        """)
        
        self.volume_value = QLabel("75")
        self.volume_value.setStyleSheet("color: #34495e; min-width: 30px;")
        
        self.volume_slider.valueChanged.connect(
            lambda v: self.volume_value.setText(str(v))
        )
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_value)
        
        # 语言设置
        language_layout = QHBoxLayout()
        language_layout.setSpacing(10)
        
        language_label = QLabel("语言:")
        language_label.setStyleSheet("color: #34495e;")
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "English", "日本語", "Español"])
        self.language_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid rgba(189, 195, 199, 150);
                border-radius: 5px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        
        # 保存设置按钮
        save_button = AnimatedButton("保存设置")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(46, 204, 113, 150);
                color: white;
            }
            QPushButton:hover {
                background-color: rgba(39, 174, 96, 200);
            }
        """)
        save_button.clicked.connect(self.save_settings)
        
        # 添加所有设置到布局
        settings_layout.addLayout(theme_layout)
        settings_layout.addLayout(volume_layout)
        settings_layout.addLayout(language_layout)
        settings_layout.addSpacing(20)
        settings_layout.addWidget(save_button, alignment=Qt.AlignCenter)
        
        settings_group.setLayout(settings_layout)
        
        # 将所有内容添加到主布局
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(settings_group)
        layout.addStretch()
    
    def select_theme_color(self):
        """选择主题颜色"""
        color = QColorDialog.getColor(QColor(52, 152, 219), self, "选择主题颜色")
        if color.isValid():
            self.theme_color_preview.setStyleSheet(f"""
                QLabel {{
                    background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                    border-radius: 15px;
                    border: 2px solid white;
                }}
            """)
    
    def save_settings(self):
        """保存设置"""
        theme_color = self.theme_color_preview.styleSheet().split("rgb(")[1].split(")")[0]
        volume = self.volume_slider.value()
        language = self.language_combo.currentText()
        
        print(f"保存设置:")
        print(f"  主题颜色: {theme_color}")
        print(f"  音量: {volume}")
        print(f"  语言: {language}")
        
        # 显示保存成功消息
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("保存成功")
        msg.setText("设置已成功保存！")
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border-radius: 10px;
            }
            QLabel {
                color: #2c3e50;
            }
            QPushButton {
                min-width: 80px;
                padding: 5px;
                border-radius: 5px;
                background-color: rgba(52, 152, 219, 150);
                color: white;
            }
            QPushButton:hover {
                background-color: rgba(41, 128, 185, 200);
            }
        """)
        msg.exec_()

class AboutPage(RoundedWidget):
    """关于界面"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 标题
        title = QLabel("关于本应用")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px 0;
            }
        """)
        
        # 应用图标
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignCenter)
        
        # 使用内置图标或检查资源是否存在
        pixmap = QPixmap(":/images/app_icon.png")
        if pixmap.isNull():  # 检查是否加载成功
            # 创建默认图标
            pixmap = QPixmap(80, 80)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setBrush(QColor(52, 152, 219))
            painter.drawEllipse(0, 0, 80, 80)
            painter.end()
        
        icon_label.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 关于文本
        about_text = QLabel("""
            <div style="color: #34495e; line-height: 1.6;">
                <p style="font-size: 16px; font-weight: bold; text-align: center;">
                    现代化UI应用示例
                </p>
                <p style="text-align: center;">
                    版本: 1.0.0<br>
                    版权所有 &copy; 2023
                </p>
                <p>
                    本应用展示了如何在PyQt5中实现现代化UI设计，包括:
                </p>
                <ul style="margin-left: 20px;">
                    <li>渐变背景和圆角边框</li>
                    <li>平滑的动画过渡效果</li>
                    <li>悬停交互反馈</li>
                    <li>现代化的配色方案</li>
                </ul>
            </div>
        """)
        about_text.setWordWrap(True)
        
        # 关闭按钮
        close_button = AnimatedButton("关闭")
        close_button.clicked.connect(lambda: QApplication.activeWindow().close())
        
        # 添加所有部件到布局
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(icon_label)
        layout.addWidget(about_text)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)
        layout.addStretch()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())