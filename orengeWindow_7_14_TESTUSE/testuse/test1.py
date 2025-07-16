import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                            QSlider, QComboBox, QColorDialog, QCheckBox, 
                            QGroupBox, QListWidget, QListWidgetItem, 
                            QMessageBox, QFrame, QLineEdit)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint, QSize
from PyQt5.QtGui import QColor, QIcon, QPainter, QBrush, QLinearGradient, QPalette, QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高级多界面应用")
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框窗口
        self.setAttribute(Qt.WA_TranslucentBackground)  # 背景透明
        
        # 创建中心部件和主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建主窗口框架（带有圆角和阴影效果）
        self.main_frame = QFrame()
        self.main_frame.setObjectName("mainFrame")
        self.main_frame.setStyleSheet("""
            #mainFrame {
                background-color: rgba(255, 255, 255, 240);
                border-radius: 15px;
                box-shadow: 0px 0px 25px rgba(0, 0, 0, 0.2);
            }
        """)
        self.main_layout.addWidget(self.main_frame)
        
        # 创建主框架内的布局
        self.frame_layout = QVBoxLayout(self.main_frame)
        
        # 创建顶部标题栏
        self.create_title_bar()
        
        # 创建内容区域
        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout(self.content_widget)
        
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
        
        # 将堆叠窗口添加到内容布局
        self.content_layout.addWidget(self.stacked_widget, 4)  # 占4份空间
        
        # 将内容区域添加到主框架布局
        self.frame_layout.addWidget(self.content_widget)
        
        # 窗口拖动相关变量
        self.draggable = True
        self.dragging_position = QPoint()
    
    def create_title_bar(self):
        """创建顶部标题栏"""
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setStyleSheet("""
            #titleBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                            stop:0 rgba(72, 126, 176, 230), 
                                            stop:1 rgba(52, 152, 219, 230));
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
            }
        """)
        title_bar.setFixedHeight(40)
        
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        
        # 应用标题
        title_label = QLabel("高级多界面应用")
        title_label.setStyleSheet("""
            color: white;
            font-size: 14px;
            font-weight: bold;
        """)
        
        # 窗口控制按钮
        self.minimize_button = QPushButton("—")
        self.maximize_button = QPushButton("□")
        self.close_button = QPushButton("×")
        
        for btn in [self.minimize_button, self.maximize_button, self.close_button]:
            btn.setFixedSize(24, 24)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 20);
                    color: white;
                    border-radius: 12px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 40);
                }
            """)
        
        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize)
        self.close_button.clicked.connect(self.close)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.minimize_button)
        title_layout.addWidget(self.maximize_button)
        title_layout.addWidget(self.close_button)
        
        self.frame_layout.addWidget(title_bar)
    
    def toggle_maximize(self):
        """切换窗口最大化/还原"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    
    def create_sidebar(self):
        """创建侧边栏菜单"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMaximumWidth(180)
        sidebar.setStyleSheet("""
            #sidebar {
                background-color: rgba(245, 245, 245, 220);
                border-bottom-left-radius: 15px;
            }
        """)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        
        # 侧边栏标题
        sidebar_title = QLabel("菜单")
        sidebar_title.setAlignment(Qt.AlignCenter)
        sidebar_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 15px;
        """)
        
        # 创建菜单列表
        self.menu_list = QListWidget()
        self.menu_list.setObjectName("menuList")
        self.menu_list.setStyleSheet("""
            QListWidget#menuList {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 5px;
                color: #2c3e50;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                            stop:0 rgba(72, 126, 176, 200), 
                                            stop:1 rgba(52, 152, 219, 200));
                color: white;
            }
            QListWidget::item:hover {
                background-color: rgba(52, 152, 219, 10);
            }
        """)
        
        # 添加菜单项
        menu_items = ["主页", "设置", "关于"]
        icons = ["🏠", "⚙️", "ℹ️"]
        
        for i, (text, icon) in enumerate(zip(menu_items, icons)):
            item = QListWidgetItem(f"{icon} {text}")
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            item.setSizeHint(QSize(0, 40))
            self.menu_list.addItem(item)
        
        # 连接列表项点击事件
        self.menu_list.currentRowChanged.connect(self.change_page)
        
        # 底部信息
        bottom_info = QLabel("© 2023 高级应用")
        bottom_info.setAlignment(Qt.AlignCenter)
        bottom_info.setStyleSheet("""
            color: #7f8c8d;
            font-size: 11px;
            margin-top: 20px;
        """)
        
        # 将所有部件添加到侧边栏布局
        sidebar_layout.addWidget(sidebar_title)
        sidebar_layout.addWidget(self.menu_list)
        sidebar_layout.addWidget(bottom_info)
        sidebar_layout.addStretch()
        
        # 将侧边栏添加到主布局
        self.content_layout.addWidget(sidebar, 1)  # 占1份空间
    
    def change_page(self, index):
        """根据选择的菜单项切换页面"""
        # 添加页面切换动画
        current_widget = self.stacked_widget.currentWidget()
        next_widget = self.stacked_widget.widget(index)
        
        if current_widget and next_widget:
            # 创建淡出动画
            fade_out = QPropertyAnimation(current_widget, b"windowOpacity")
            fade_out.setDuration(300)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.setEasingCurve(QEasingCurve.InOutQuad)
            
            # 创建淡入动画
            fade_in = QPropertyAnimation(next_widget, b"windowOpacity")
            fade_in.setDuration(300)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.setEasingCurve(QEasingCurve.InOutQuad)
            
            # 按顺序执行动画
            fade_out.finished.connect(lambda: self.stacked_widget.setCurrentIndex(index))
            fade_out.finished.connect(fade_in.start)
            
            # 启动淡出动画
            fade_out.start()
    
    # 以下是实现窗口拖动的代码
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.draggable:
            self.dragging = True
            self.dragging_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self.dragging:
            self.move(event.globalPos() - self.dragging_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self.dragging = False
        event.accept()

class MainPage(QWidget):
    """主界面"""
    def __init__(self):
        super().__init__()
        self.setObjectName("mainPage")
        self.setStyleSheet("""
            #mainPage {
                background-color: rgba(255, 255, 255, 0);
                border-bottom-right-radius: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 顶部欢迎区域
        welcome_widget = QWidget()
        welcome_widget.setObjectName("welcomeWidget")
        welcome_widget.setStyleSheet("""
            #welcomeWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                            stop:0 rgba(72, 126, 176, 100), 
                                            stop:1 rgba(52, 152, 219, 100));
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        welcome_layout = QHBoxLayout(welcome_widget)
        
        welcome_text = QLabel("""
            <h2>欢迎使用高级多界面应用</h2>
            <p>这是一个功能丰富、界面美观的PyQt5应用示例，展示了现代GUI设计的最佳实践。</p>
        """)
        welcome_text.setWordWrap(True)
        welcome_text.setStyleSheet("color: #2c3e50;")
        
        welcome_icon = QLabel("🚀")
        welcome_icon.setStyleSheet("font-size: 60px;")
        
        welcome_layout.addWidget(welcome_text)
        welcome_layout.addStretch()
        welcome_layout.addWidget(welcome_icon)
        
        layout.addWidget(welcome_widget)
        layout.addSpacing(30)
        
        # 功能卡片区域
        cards_layout = QHBoxLayout()
        
        # 卡片1: 随机数生成器
        card1 = QWidget()
        card1.setObjectName("card1")
        card1.setStyleSheet("""
            #card1 {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            #card1:hover {
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
                transform: translateY(-5px);
            }
        """)
        
        card1_layout = QVBoxLayout(card1)
        card1_title = QLabel("随机数生成器")
        card1_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        
        self.random_number_label = QLabel("点击按钮生成随机数")
        self.random_number_label.setAlignment(Qt.AlignCenter)
        self.random_number_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #3498db; margin: 15px 0;")
        
        card1_button = QPushButton("生成随机数")
        card1_button.setObjectName("card1Button")
        card1_button.setStyleSheet("""
            #card1Button {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            #card1Button:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #2980b9, stop:1 #2073a8);
            }
        """)
        card1_button.clicked.connect(self.generate_random_number)
        
        card1_layout.addWidget(card1_title)
        card1_layout.addWidget(self.random_number_label)
        card1_layout.addWidget(card1_button)
        
        # 卡片2: 进度展示
        card2 = QWidget()
        card2.setObjectName("card2")
        card2.setStyleSheet("""
            #card2 {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            #card2:hover {
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
                transform: translateY(-5px);
            }
        """)
        
        card2_layout = QVBoxLayout(card2)
        card2_title = QLabel("进度展示")
        card2_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        
        progress_label = QLabel("操作进度")
        progress_label.setStyleSheet("color: #7f8c8d;")
        
        self.progress_bar = QLabel()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setStyleSheet("""
            #progressBar {
                background-color: #ecf0f1;
                border-radius: 10px;
                margin: 10px 0;
            }
        """)
        
        self.progress_value = QLabel()
        self.progress_value.setFixedHeight(20)
        self.progress_value.setObjectName("progressValue")
        self.progress_value.setStyleSheet("""
            #progressValue {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #2ecc71, stop:1 #27ae60);
                border-radius: 10px;
                width: 45%;
            }
        """)
        
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_value)
        progress_layout.addStretch()
        
        self.progress_percent = QLabel("45%")
        self.progress_percent.setAlignment(Qt.AlignCenter)
        self.progress_percent.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        card2_button = QPushButton("更新进度")
        card2_button.setObjectName("card2Button")
        card2_button.setStyleSheet("""
            #card2Button {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #2ecc71, stop:1 #27ae60);
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            #card2Button:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #27ae60, stop:1 #219a52);
            }
        """)
        card2_button.clicked.connect(self.update_progress)
        
        card2_layout.addWidget(card2_title)
        card2_layout.addWidget(progress_label)
        card2_layout.addWidget(self.progress_bar)
        card2_layout.addLayout(progress_layout)
        card2_layout.addWidget(self.progress_percent)
        card2_layout.addWidget(card2_button)
        
        # 添加卡片到布局
        cards_layout.addWidget(card1)
        cards_layout.addSpacing(20)
        cards_layout.addWidget(card2)
        
        layout.addLayout(cards_layout)
        layout.addStretch()
    
    def generate_random_number(self):
        """生成随机数"""
        import random
        number = random.randint(1, 100)
        self.random_number_label.setText(str(number))
        
        # 添加数字变化动画
        animation = QPropertyAnimation(self.random_number_label, b"fontSize")
        animation.setDuration(300)
        animation.setStartValue(24)
        animation.setEndValue(36)
        animation.setEasingCurve(QEasingCurve.OutElastic)
        animation.start()
        
        # 恢复字体大小
        def restore_font_size():
            self.random_number_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: #3498db; margin: 15px 0;")
        
        QTimer.singleShot(300, restore_font_size)
    
    def update_progress(self):
        """更新进度条"""
        import random
        progress = random.randint(10, 100)
        self.progress_percent.setText(f"{progress}%")
        self.progress_value.setStyleSheet(f"""
            #progressValue {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #2ecc71, stop:1 #27ae60);
                border-radius: 10px;
                width: {progress}%;
            }
        """)

class SettingsPage(QWidget):
    """设置界面"""
    def __init__(self):
        super().__init__()
        self.setObjectName("settingsPage")
        self.setStyleSheet("""
            #settingsPage {
                background-color: rgba(255, 255, 255, 0);
                border-bottom-right-radius: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题区域
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        
        title = QLabel("应用设置")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        layout.addWidget(title_widget)
        layout.addSpacing(20)
        
        # 设置卡片
        settings_card = QWidget()
        settings_card.setObjectName("settingsCard")
        settings_card.setStyleSheet("""
            #settingsCard {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.05);
            }
        """)
        
        settings_layout = QVBoxLayout(settings_card)
        
        # 主题颜色设置
        theme_group = QGroupBox("主题设置")
        theme_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ecf0f1;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        theme_layout = QVBoxLayout(theme_group)
        
        # 主题颜色选择
        color_layout = QHBoxLayout()
        color_label = QLabel("主题颜色:")
        color_label.setStyleSheet("color: #7f8c8d;")
        
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setObjectName("colorPreview")
        self.color_preview.setStyleSheet("""
            #colorPreview {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #3498db, stop:1 #2980b9);
                border-radius: 5px;
            }
        """)
        
        color_button = QPushButton("选择颜色")
        color_button.setObjectName("colorButton")
        color_button.setStyleSheet("""
            #colorButton {
                background-color: #ecf0f1;
                color: #2c3e50;
                border-radius: 5px;
                padding: 5px 10px;
            }
            #colorButton:hover {
                background-color: #e0e7ea;
            }
        """)
        color_button.clicked.connect(self.select_color)
        
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(color_button)
        color_layout.addStretch()
        
        # 主题模式选择
        theme_mode_layout = QHBoxLayout()
        theme_mode_label = QLabel("主题模式:")
        theme_mode_label.setStyleSheet("color: #7f8c8d;")
        
        self.theme_mode_combo = QComboBox()
        self.theme_mode_combo.addItems(["浅色模式", "深色模式", "跟随系统"])
        self.theme_mode_combo.setObjectName("themeModeCombo")
        self.theme_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #ecf0f1;
                color: #2c3e50;
                border-radius: 5px;
                padding: 5px;
                border: none;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        
        theme_mode_layout.addWidget(theme_mode_label)
        theme_mode_layout.addWidget(self.theme_mode_combo)
        theme_mode_layout.addStretch()
        
        theme_layout.addLayout(color_layout)
        theme_layout.addLayout(theme_mode_layout)
        
        # 应用设置
        app_group = QGroupBox("应用设置")
        app_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ecf0f1;
                border-radius: 10px;
                margin-top: 20px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        app_layout = QVBoxLayout(app_group)
        
        # 启动设置
        startup_layout = QHBoxLayout()
        startup_label = QLabel("开机自启动:")
        startup_label.setStyleSheet("color: #7f8c8d;")
        
        self.startup_checkbox = QCheckBox()
        self.startup_checkbox.setObjectName("startupCheckbox")
        self.startup_checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                background-color: #ecf0f1;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
            }
            QCheckBox::indicator:checked::before {
                content: "✓";
                color: white;
                position: absolute;
                left: 4px;
                top: 1px;
            }
        """)
        
        startup_layout.addWidget(startup_label)
        startup_layout.addWidget(self.startup_checkbox)
        startup_layout.addStretch()
        
        # 自动更新设置
        update_layout = QHBoxLayout()
        update_label = QLabel("自动更新:")
        update_label.setStyleSheet("color: #7f8c8d;")
        
        self.update_checkbox = QCheckBox()
        self.update_checkbox.setObjectName("updateCheckbox")
        self.update_checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                background-color: #ecf0f1;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
            }
            QCheckBox::indicator:checked::before {
                content: "✓";
                color: white;
                position: absolute;
                left: 4px;
                top: 1px;
            }
        """)
        
        update_layout.addWidget(update_label)
        update_layout.addWidget(self.update_checkbox)
        update_layout.addStretch()
        
        # 音量设置
        volume_layout = QHBoxLayout()
        volume_label = QLabel("音量:")
        volume_label.setStyleSheet("color: #7f8c8d;")
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(75)
        self.volume_slider.setObjectName("volumeSlider")
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
        """)
        
        self.volume_value = QLabel("75")
        self.volume_value.setFixedWidth(30)
        self.volume_value.setStyleSheet("color: #2c3e50;")
        
        self.volume_slider.valueChanged.connect(lambda: self.volume_value.setText(str(self.volume_slider.value())))
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_value)
        
        app_layout.addLayout(startup_layout)
        app_layout.addLayout(update_layout)
        app_layout.addLayout(volume_layout)
        
        # 保存设置按钮
        save_button = QPushButton("保存设置")
        save_button.setObjectName("saveButton")
        save_button.setStyleSheet("""
            #saveButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                margin-top: 20px;
            }
            #saveButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #2980b9, stop:1 #2073a8);
            }
        """)
        save_button.clicked.connect(self.save_settings)
        
        settings_layout.addWidget(theme_group)
        settings_layout.addWidget(app_group)
        settings_layout.addWidget(save_button, alignment=Qt.AlignCenter)
        
        layout.addWidget(settings_card)
        layout.addStretch()
    
    def select_color(self):
        """选择主题颜色"""
        color = QColorDialog.getColor(QColor(52, 152, 219), self, "选择主题颜色")
        if color.isValid():
            # 更新预览
            self.color_preview.setStyleSheet(f"""
                #colorPreview {{
                    background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                    border-radius: 5px;
                }}
            """)
    
    def save_settings(self):
        """保存设置"""
        # 模拟保存设置
        theme_color = self.color_preview.styleSheet().split("rgb(")[1].split(")")[0]
        theme_mode = self.theme_mode_combo.currentText()
        startup = self.startup_checkbox.isChecked()
        update = self.update_checkbox.isChecked()
        volume = self.volume_slider.value()
        
        print(f"保存设置:")
        print(f"  主题颜色: {theme_color}")
        print(f"  主题模式: {theme_mode}")
        print(f"  开机自启动: {startup}")
        print(f"  自动更新: {update}")
        print(f"  音量: {volume}")
        
        # 显示保存成功消息
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("保存成功")
        msg_box.setText("设置已成功保存！")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        
        # 设置消息框样式
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border-radius: 10px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        msg_box.exec_()

class AboutPage(QWidget):
    """关于界面"""
    def __init__(self):
        super().__init__()
        self.setObjectName("aboutPage")
        self.setStyleSheet("""
            #aboutPage {
                background-color: rgba(255, 255, 255, 0);
                border-bottom-right-radius: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题区域
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        
        title = QLabel("关于应用")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        layout.addWidget(title_widget)
        layout.addSpacing(20)
        
        # 关于卡片
        about_card = QWidget()
        about_card.setObjectName("aboutCard")
        about_card.setStyleSheet("""
            #aboutCard {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.05);
            }
        """)
        
        about_layout = QVBoxLayout(about_card)
        
        # 应用信息
        info_layout = QHBoxLayout()
        
        app_icon = QLabel("✨")
        app_icon.setStyleSheet("font-size: 80px;")
        
        app_info = QLabel("""
            <h2>高级多界面应用</h2>
            <p>版本 1.0.0</p>
            <p>一个功能丰富、界面美观的PyQt5应用示例</p>
            <p>© 2023 开发者团队</p>
        """)
        app_info.setStyleSheet("color: #2c3e50;")
        
        info_layout.addWidget(app_icon)
        info_layout.addSpacing(20)
        info_layout.addWidget(app_info)
        info_layout.addStretch()
        
        # 应用描述
        description = QLabel("""
            <p>本应用展示了如何使用PyQt5创建现代化的桌面应用程序，包括：</p>
            <ul>
                <li>多界面切换系统</li>
                <li>自定义UI设计和动画效果</li>
                <li>响应式布局和交互元素</li>
                <li>渐变配色和圆润边框设计</li>
                <li>透明效果和阴影层次</li>
            </ul>
            <p>您可以自由修改和扩展此应用，用于学习或实际项目开发。</p>
        """)
        description.setWordWrap(True)
        description.setStyleSheet("color: #2c3e50;")
        
        # 许可证信息
        license_info = QLabel("""
            <p>本应用采用MIT许可证发布：</p>
            <p>Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:</p>
            <p>The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.</p>
        """)
        license_info.setWordWrap(True)
        license_info.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        
        about_layout.addLayout(info_layout)
        about_layout.addSpacing(20)
        about_layout.addWidget(description)
        about_layout.addSpacing(20)
        about_layout.addWidget(license_info)
        
        layout.addWidget(about_card)
        layout.addStretch()

# 导入QTimer用于动画效果
from PyQt5.QtCore import QTimer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())