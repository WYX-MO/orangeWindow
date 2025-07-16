import sys
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine
import math

class CalculatorBackend(QObject):
    """计算器后端逻辑"""
    
    def __init__(self):
        super().__init__()
        self.current_expression = ""  # 当前表达式
        self.last_result = ""         # 上次计算结果
        self.is_new_calculation = True  # 是否开始新计算
    
    # 向QML发送更新的信号
    displayTextChanged = pyqtSignal(str)
    
    @pyqtSlot(str)
    def buttonClicked(self, button_text):
        """处理按钮点击事件"""
        # 数字和小数点
        if button_text.isdigit() or button_text == '.':
            self.handle_number(button_text)
        
        # 运算符
        elif button_text in ['+', '-', '*', '/', '%']:
            self.handle_operator(button_text)
        
        # 等号
        elif button_text == '=':
            self.calculate()
        
        # 清除
        elif button_text == 'C':
            self.clear_all()
        
        # 正负号
        elif button_text == '+/-':
            self.toggle_sign()
        
        # 退格
        elif button_text == '⌫':
            self.backspace()
        
        # 平方根
        elif button_text == '√':
            self.calculate_sqrt()
        
        # 倒数
        elif button_text == '1/x':
            self.calculate_reciprocal()
        
        # 百分号
        elif button_text == '%':
            self.calculate_percentage()
        
        # 发送更新后的显示文本到QML
        self.displayTextChanged.emit(self.get_display_text())
    
    def handle_number(self, number):
        """处理数字输入"""
        if self.is_new_calculation:
            self.current_expression = number
            self.is_new_calculation = False
        else:
            # 避免多个小数点
            if number == '.' and '.' in self.current_expression.split()[-1]:
                return
            self.current_expression += number
    
    def handle_operator(self, operator):
        """处理运算符"""
        if self.is_new_calculation and self.last_result:
            self.current_expression = self.last_result + " " + operator + " "
            self.is_new_calculation = False
        else:
            # 如果表达式末尾已有运算符，替换它
            if self.current_expression and self.current_expression[-1] in ['+', '-', '*', '/', '%']:
                self.current_expression = self.current_expression[:-1] + operator
            else:
                self.current_expression += " " + operator + " "
    
    def calculate(self):
        """计算表达式结果"""
        try:
            # 替换百分比为小数
            expr = self.current_expression.replace('%', '/100')
            result = eval(expr)
            
            # 格式化结果
            if result == int(result):
                self.last_result = str(int(result))
            else:
                self.last_result = str(result)
            
            self.current_expression = self.last_result
            self.is_new_calculation = True
        except Exception as e:
            self.current_expression = "Error"
            self.is_new_calculation = True
    
    def clear_all(self):
        """清除所有内容"""
        self.current_expression = ""
        self.is_new_calculation = True
    
    def toggle_sign(self):
        """切换正负号"""
        if not self.current_expression or self.current_expression == "0":
            return
        
        if self.is_new_calculation:
            self.current_expression = self.last_result
        
        # 如果是表达式的开始，直接添加负号
        if not self.current_expression.startswith('-'):
            self.current_expression = '-' + self.current_expression
        else:
            self.current_expression = self.current_expression[1:]
    
    def backspace(self):
        """删除最后一个字符"""
        if self.current_expression:
            self.current_expression = self.current_expression[:-1]
    
    def calculate_sqrt(self):
        """计算平方根"""
        try:
            value = float(self.current_expression)
            if value < 0:
                self.current_expression = "Error"
            else:
                result = math.sqrt(value)
                self.current_expression = str(result)
            self.is_new_calculation = True
        except:
            self.current_expression = "Error"
            self.is_new_calculation = True
    
    def calculate_reciprocal(self):
        """计算倒数"""
        try:
            value = float(self.current_expression)
            if value == 0:
                self.current_expression = "Error"
            else:
                result = 1 / value
                self.current_expression = str(result)
            self.is_new_calculation = True
        except:
            self.current_expression = "Error"
            self.is_new_calculation = True
    
    def calculate_percentage(self):
        """计算百分比"""
        try:
            value = float(self.current_expression)
            self.current_expression = str(value / 100)
        except:
            pass
    
    def get_display_text(self):
        """获取要显示的文本，添加千位分隔符"""
        if not self.current_expression:
            return "0"
        
        # 如果是错误信息，直接返回
        if self.current_expression == "Error":
            return self.current_expression
        
        # 尝试格式化数字
        try:
            # 如果是小数，分开处理整数部分和小数部分
            if '.' in self.current_expression:
                integer_part, decimal_part = self.current_expression.split('.')
                formatted_integer = "{:,}".format(int(integer_part))
                return f"{formatted_integer}.{decimal_part}"
            else:
                return "{:,}".format(int(self.current_expression))
        except:
            return self.current_expression

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    
    # 创建计算器后端实例
    calculator = CalculatorBackend()
    
    # 创建QML引擎
    engine = QQmlApplicationEngine()
    
    # 将计算器后端实例注入到QML环境
    engine.rootContext().setContextProperty("calculator", calculator)
    
    # 加载QML文件
    engine.load("codes\\Projecion\\orengeWindow_7_14\\testuse\\calculator.qml")
    
    # 检查是否成功加载
    if not engine.rootObjects():
        sys.exit(-1)
    
    sys.exit(app.exec_())