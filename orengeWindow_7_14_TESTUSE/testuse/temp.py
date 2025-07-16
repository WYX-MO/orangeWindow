import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                            QSlider, QComboBox, QColorDialog, QCheckBox, 
                            QGroupBox, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多界面应用")
        self.setGeometry(100, 100, 800, 500)
        
        # 创建中心部件和主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
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
        self.main_layout.addWidget(self.stacked_widget, 4)  # 占4份空间
    
    def create_sidebar(self):
        """创建侧边栏菜单"""
        sidebar = QWidget()
        sidebar.setMaximumWidth(150)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # 创建菜单列表
        self.menu_list = QListWidget()
        
        # 添加菜单项
        menu_items = ["主页", "设置", "关于"]
        for item_text in menu_items:
            item = QListWidgetItem(item_text)
            self.menu_list.addItem(item)
        
        # 连接列表项点击事件
        self.menu_list.currentRowChanged.connect(self.change_page)
        
        # 将菜单添加到侧边栏布局
        sidebar_layout.addWidget(self.menu_list)
        sidebar_layout.addStretch()
        
        # 将侧边栏添加到主布局
        self.main_layout.addWidget(sidebar, 1)  # 占1份空间
    
    def change_page(self, index):
        """根据选择的菜单项切换页面"""
        self.stacked_widget.setCurrentIndex(index)

class MainPage(QWidget):
    """主界面"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        title = QLabel("欢迎来到主界面")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        content = QLabel("这是应用的主界面，您可以在这里查看主要内容和快速访问常用功能。")
        content.setAlignment(Qt.AlignCenter)
        content.setWordWrap(True)
        
        random_button = QPushButton("生成随机数")
        random_button.clicked.connect(self.generate_random_number)
        
        self.result_label = QLabel("结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignCenter)
        
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(content)
        layout.addSpacing(20)
        layout.addWidget(random_button)
        layout.addWidget(self.result_label)
        layout.addStretch()
    
    def generate_random_number(self):
        import random
        number = random.randint(1, 100)
        self.result_label.setText(f"随机数: {number}")

class SettingsPage(QWidget):
    """设置界面"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        title = QLabel("设置")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        # 创建设置内容
        settings_group = QGroupBox("应用设置")
        settings_layout = QVBoxLayout()
        
        # 主题颜色设置
        theme_layout = QHBoxLayout()
        theme_label = QLabel("主题颜色:")
        self.theme_color_preview = QLabel()
        self.theme_color_preview.setFixedSize(30, 30)
        self.theme_color_preview.setStyleSheet("background-color: rgb(52, 152, 219)")
        
        theme_button = QPushButton("选择颜色")
        theme_button.clicked.connect(self.select_theme_color)
        
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_color_preview)
        theme_layout.addWidget(theme_button)
        theme_layout.addStretch()
        
        # 音量设置
        volume_layout = QHBoxLayout()
        volume_label = QLabel("音量:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(75)
        
        self.volume_value = QLabel("75")
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_value)
        
        # 语言设置
        language_layout = QHBoxLayout()
        language_label = QLabel("语言:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "English", "日本語", "Español"])
        
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        
        # 保存设置按钮
        save_button = QPushButton("保存设置")
        save_button.setStyleSheet("background-color: #2ecc71; color: white;")
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
            self.theme_color_preview.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")
    
    def save_settings(self):
        """保存设置"""
        # 实际应用中这里会将设置保存到文件或数据库
        theme_color = self.theme_color_preview.styleSheet().split("rgb(")[1].split(")")[0]
        volume = self.volume_slider.value()
        language = self.language_combo.currentText()
        
        print(f"保存设置:")
        print(f"  主题颜色: {theme_color}")
        print(f"  音量: {volume}")
        print(f"  语言: {language}")
        
        # 显示保存成功消息
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "保存成功", "设置已成功保存！")

class AboutPage(QWidget):
    """关于界面"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        title = QLabel("关于本应用")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        about_text = QLabel("""
        <p>这是一个使用PyQt5创建的多界面应用示例。</p>
        <p>版本: 1.0.0</p>
        <p>版权所有 &copy; 2023</p>
        <p>本应用展示了如何在PyQt5中实现界面切换功能，包括：</p>
        <ul>
            <li>使用QStackedWidget管理多个界面</li>
            <li>创建侧边栏导航菜单</li>
            <li>实现界面间的数据传递</li>
            <li>自定义界面样式</li>
        </ul>
        """)
        about_text.setAlignment(Qt.AlignLeft)
        about_text.setWordWrap(True)
        
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(about_text)
        layout.addStretch()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())