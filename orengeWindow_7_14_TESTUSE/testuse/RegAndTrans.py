import sys
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, 
                            QComboBox, QColorDialog, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("设置窗口")
        self.setGeometry(100, 100, 600, 400)
        
        # 设置默认值
        self.theme_color = QColor(52, 152, 219)  # 蓝色
        self.number1 = random.randint(1, 100)
        self.number2 = random.randint(1, 100)
        self.number3 = random.randint(1, 100)
        self.boolean_setting = False
        self.numeric_setting = 50
        self.enum_setting = "选项1"
        
        # 创建中心部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建主题颜色选择器
        self.create_theme_color_section()
        
        # 创建数字显示区域
        self.create_numbers_section()
        
        # 创建设置选项区域
        self.create_settings_section()
        
        # 应用初始主题
        self.apply_theme()
    
    def create_theme_color_section(self):
        """创建主题颜色选择区域"""
        theme_group = QGroupBox("主题颜色")
        theme_layout = QHBoxLayout()
        
        self.theme_label = QLabel("当前主题:")
        self.theme_color_preview = QLabel()
        self.theme_color_preview.setFixedSize(30, 30)
        self.theme_color_preview.setStyleSheet(f"background-color: rgb({self.theme_color.red()}, {self.theme_color.green()}, {self.theme_color.blue()})")
        
        self.theme_button = QPushButton("选择颜色")
        self.theme_button.clicked.connect(self.select_theme_color)
        
        theme_layout.addWidget(self.theme_label)
        theme_layout.addWidget(self.theme_color_preview)
        theme_layout.addWidget(self.theme_button)
        theme_layout.addStretch()
        
        theme_group.setLayout(theme_layout)
        self.main_layout.addWidget(theme_group)
    
    def create_numbers_section(self):
        """创建数字显示区域"""
        numbers_group = QGroupBox("数字显示")
        numbers_layout = QHBoxLayout()
        
        self.number1_label = QLabel(str(self.number1))
        self.number2_label = QLabel(str(self.number2))
        self.number3_label = QLabel(str(self.number3))
        
        for label in [self.number1_label, self.number2_label, self.number3_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 24px; font-weight: bold;")
            numbers_layout.addWidget(label)
        
        numbers_group.setLayout(numbers_layout)
        self.main_layout.addWidget(numbers_group)
    
    def create_settings_section(self):
        """创建设置选项区域"""
        settings_group = QGroupBox("设置选项")
        settings_layout = QVBoxLayout()
        
        # 布尔设置
        boolean_layout = QHBoxLayout()
        self.boolean_label = QLabel("布尔设置:")
        self.boolean_checkbox = QCheckBox()
        self.boolean_checkbox.setChecked(self.boolean_setting)
        self.boolean_checkbox.stateChanged.connect(self.update_boolean_setting)
        
        boolean_layout.addWidget(self.boolean_label)
        boolean_layout.addWidget(self.boolean_checkbox)
        boolean_layout.addStretch()
        
        # 数值设置
        numeric_layout = QHBoxLayout()
        self.numeric_label = QLabel("数值设置:")
        self.numeric_value = QLabel(str(self.numeric_setting))
        self.numeric_slider = QSlider(Qt.Horizontal)
        self.numeric_slider.setRange(0, 100)
        self.numeric_slider.setValue(self.numeric_setting)
        self.numeric_slider.valueChanged.connect(self.update_numeric_setting)
        
        numeric_layout.addWidget(self.numeric_label)
        numeric_layout.addWidget(self.numeric_slider)
        numeric_layout.addWidget(self.numeric_value)
        
        # 枚举设置
        enum_layout = QHBoxLayout()
        self.enum_label = QLabel("枚举设置:")
        self.enum_combo = QComboBox()
        self.enum_combo.addItems(["选项1", "选项2", "选项3", "选项4"])
        self.enum_combo.setCurrentText(self.enum_setting)
        self.enum_combo.currentTextChanged.connect(self.update_enum_setting)
        
        enum_layout.addWidget(self.enum_label)
        enum_layout.addWidget(self.enum_combo)
        enum_layout.addStretch()
        
        # 添加所有设置到主布局
        settings_layout.addLayout(boolean_layout)
        settings_layout.addLayout(numeric_layout)
        settings_layout.addLayout(enum_layout)
        
        settings_group.setLayout(settings_layout)
        self.main_layout.addWidget(settings_group)
    
    def select_theme_color(self):
        """选择主题颜色"""
        color = QColorDialog.getColor(self.theme_color, self, "选择主题颜色")
        if color.isValid():
            self.theme_color = color
            self.theme_color_preview.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")
            self.apply_theme()
    
    def apply_theme(self):
        """应用主题颜色"""
        # 获取RGB值
        r, g, b = self.theme_color.red(), self.theme_color.green(), self.theme_color.blue()
        
        # 应用到标题栏和按钮
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: rgb({r}, {g}, {b});
            }}
            QPushButton {{
                background-color: rgb({max(r-20, 0)}, {max(g-20, 0)}, {max(b-20, 0)});
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: rgb({max(r-30, 0)}, {max(g-30, 0)}, {max(b-30, 0)});
            }}
            QGroupBox {{
                border: 1px solid rgb({r-40}, {g-40}, {b-40});
                border-radius: 5px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }}
        """)
    
    def update_boolean_setting(self, state):
        """更新布尔设置"""
        self.boolean_setting = state == Qt.Checked
        print(f"布尔设置已更新为: {self.boolean_setting}")
    
    def update_numeric_setting(self, value):
        """更新数值设置"""
        self.numeric_setting = value
        self.numeric_value.setText(str(value))
        print(f"数值设置已更新为: {value}")
    
    def update_enum_setting(self, value):
        """更新枚举设置"""
        self.enum_setting = value
        print(f"枚举设置已更新为: {value}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.show()
    sys.exit(app.exec_())