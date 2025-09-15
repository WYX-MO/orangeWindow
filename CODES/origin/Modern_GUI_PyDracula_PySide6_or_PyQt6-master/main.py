
import sys
import os
import platform

# 导入 GUI和模块及组件
# ///////////////////////////////////////////////////////////////
from modules import *
from widgets import *
os.environ["QT_FONT_DPI"] = "96"  # 解决高DPI和缩放比例超过100%的问题

# 设置为全局组件
# ///////////////////////////////////////////////////////////////
widgets = None

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.useCustomTheme = False
        self.absPath = None
        # 设置为全局组件
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # 使用自定义标题栏 | 对于MAC或LINUX系统请设为"False"
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # 应用名称
        # ///////////////////////////////////////////////////////////////
        title = "floatwindows"
        description = "飘窗"
        # 应用文本
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # 切换菜单
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # 设置UI定义
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # 表格组件参数
        # ///////////////////////////////////////////////////////////////
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 按钮点击事件
        # ///////////////////////////////////////////////////////////////

        # 左侧菜单
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)

        # 左侧额外面板
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # 右侧额外面板
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # 显示应用
        # ///////////////////////////////////////////////////////////////
        self.show()

        # 设置自定义主题
        # ///////////////////////////////////////////////////////////////
        if getattr(sys ,'frozen',False):
            absPath = os.path.dirname(os.path.abspath(sys.executable))
        elif __file__:
            absPath = os.path.dirname(os.path.abspath(__file__))

        useCustomTheme = False
        self.useCustomTheme = useCustomTheme
        self.absPath = absPath
        themeFile = "themes\py_dracula_light.qss"

        # 设置主题和技巧
        if useCustomTheme:
            # 加载并应用样式
            UIFunctions.theme(self, themeFile, True)

            # 设置技巧
            AppFunctions.setThemeHack(self)

        # 设置主页和选择菜单
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))


    # 按钮点击事件
    # 在此处添加按钮点击的处理函数
    # ///////////////////////////////////////////////////////////////
    def buttonClick(self):
        # 获取点击的按钮
        btn = self.sender()
        btnName = btn.objectName()

        # 显示主页
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # 显示组件页
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # 显示新页面
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page)  # 设置页面
            UIFunctions.resetStyle(self, btnName)  # 重置其他选中的按钮
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # 选择菜单

        if btnName == "btn_save":
            print("保存按钮被点击!")

        # 打印按钮名称
        print(f'按钮 "{btnName}" 被点击!')


    # 尺寸调整事件
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # 更新尺寸调整控件
        UIFunctions.resize_grips(self)

    # 鼠标点击事件
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # 设置窗口拖动位置
        self.dragPos = event.globalPos()

        # 打印鼠标事件
        if event.buttons() == Qt.LeftButton:
            print('鼠标点击: 左键点击')
        if event.buttons() == Qt.RightButton:
            print('鼠标点击: 右键点击')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec_())