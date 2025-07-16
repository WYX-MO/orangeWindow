import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, 
                             QVBoxLayout, QLabel, QFileDialog)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QScreen
from PyQt5.QtGui import QImage  # 确保已导入QImage类
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal 
from numba.cuda.cudadrv.driver import AutoFreePointer
import pytesseract
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv  # 新增：用于加载环境变量
import sys
import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QHBoxLayout, QVBoxLayout, QFrame, QMessageBox, QLabel)
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QCursor, QScreen, QImage, QPixmap
from numba.cuda.cudadrv.driver import AutoFreePointer
import pytesseract
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv  # 新增：用于加载环境变量
import sys
import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QHBoxLayout, QVBoxLayout, QFrame, QMessageBox, QLabel)
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QCursor, QScreen, QImage, QPixmap


image_path = r"E:\pyLearn\imgs\comic\\1 (3).png"
tesseract_path = r"E:\pyLearn\reses\tesseract.exe"#OCR路径
BLANK = 1
GAUSSIAN = 2
# 设置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 加载环境变量（如果有）
load_dotenv()
DEEPSEEK_API_KEY = "sk-2b814ccebc6e4d90b76ef40d2dd36f83"  # API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1" # API URL





class ScreenshotTool(QWidget):
    # region_selection_finished = pyqtSignal()  # 新增这一行
    x = 0
    y = 0
    w = 0
    h = 0

    def __init__(self):
        super().__init__()
        self.initUI()
        self.screenshot_windows=[]
        # 新增标志位：标记区域选择是否完成
        self.region_selection_done = False  # 初始为False

        self.setWindowTitle("Windows区域截图工具")
        self.setGeometry(100, 100, 600, 400)
        
        # 截图相关变量
        self.screen = None
        self.full_screenshot = None
        self.selected_region = None
        self.processed_cv_image = None  # 存储处理后的OpenCV图像
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # 按钮区域
        btn_layout = QVBoxLayout()
        
        self.btn_full = QPushButton("1. 获取全屏截图")
        self.btn_full.clicked.connect(self.capture_fullscreen)
        
        self.btn_select = QPushButton("2. 选择截图区域")
        self.btn_select.clicked.connect(self.select_region)
        self.btn_select.setEnabled(False)
        
        self.btn_crop = QPushButton("3. 截取选定区域")
        self.btn_crop.clicked.connect(self.crop_selected_region)
        self.btn_crop.setEnabled(False)
        
        self.btn_process = QPushButton("4. 处理并返回OpenCV格式")
        self.btn_process.clicked.connect(self.process_and_return_opencv)
        self.btn_process.setEnabled(False)


        self.btn_1 = QPushButton("==一键处理.beta==")
        self.btn_1.clicked.connect(self.one)
        
        btn_layout.addWidget(self.btn_full)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_crop)
        btn_layout.addWidget(self.btn_process)
        btn_layout.addWidget(self.btn_1)
        
        # 显示区域
        self.label = QLabel("截图预览区域")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #aaa;
                min-height: 300px;
            }
        """)
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

    # def one(self):
    #     self.capture_fullscreen()
    #     self.select_region()
    #     self.crop_selected_region()
    #     self.process_and_return_opencv()
    #     print(self.x,self.y)
    def one(self):
        self.region_selection_done = False  # 重置标志位
        self.capture_fullscreen()
        self.select_region()  # 弹出选择窗口
        
        # 循环等待区域选择完成（同时处理事件循环）
        while not self.region_selection_done:
            QApplication.processEvents()  # 允许处理用户操作（如鼠标选择）
        
        # 选择完成后执行后续步骤
        self.show()
        self.crop_selected_region()
        cv_img = self.process_and_return_opencv()
        cv2.imshow("111",cv_img)
        print("当前坐标：", self.x, self.y)  # 最新坐标
        self.translate(image_path,cv_img)

    def translate(self,image_path,cv_img):
        
            RAT = RegAndTranser(image_path)
            # 4. 示例处理：转换为灰度图
            # processed_cv1 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            # print("or")
            # print(processed_cv1.shape)
            processed_cv = RAT.process(cv_img)
            # 正确的resize语法：使用元组指定(宽度, 高度)
            processed_cv = cv2.resize(processed_cv, (cv_img.shape[1],cv_img.shape[0]))
            processed_cv = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2GRAY)

            # print("af")
            # print(processed_cv.shape)
            cv2.imshow("666",processed_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            width = self.w
            height = self.h
            # 5. 将处理后的OpenCV图像转回QImage
            bytes_per_line = 1 * width  # 灰度图每个像素1字节
            qimage_processed = QImage(processed_cv.data, width, height, 
                                    bytes_per_line, QImage.Format_Grayscale8)
            
            # 6. 保存处理后的图片
            # processed_path = os.path.join(desktop_path, "processed_screenshot.png")
            # qimage_processed.save(processed_path, "PNG")
            
            # 7. 将QImage转回QPixmap用于显示
            pixmap_processed = QPixmap.fromImage(qimage_processed)
            #============================END====================================

            # 创建独立窗口（不关联父窗口）
            screenshot_window = ScreenshotWindow(pixmap_processed,self.x,self.y)
            screenshot_window.show()
            
            # 管理窗口列表
            self.screenshot_windows.append(screenshot_window)

    def capture_fullscreen(self):
        """捕获全屏截图"""
        self.hide()
        QApplication.processEvents()
        
        # 使用PyQt5的QScreen.grabWindow方法获取全屏截图
        self.screen = QApplication.primaryScreen()
        self.full_screenshot = self.screen.grabWindow(0)
        
        # # 显示全屏截图（缩放到适合窗口大小）
        # self.label.setPixmap(self.full_screenshot.scaled(
        #     self.label.width(), self.label.height(),
        #     Qt.KeepAspectRatio, Qt.SmoothTransformation
        # ))
        
        self.btn_select.setEnabled(True)
        
    
    def select_region(self):
        """在全屏截图上选择区域"""
        if self.full_screenshot is None:
            return
            
        # 创建区域选择窗口
        self.selector = RegionSelector(self.full_screenshot)
        self.selector.region_selected.connect(self.handle_region_selected)
        self.selector.showFullScreen()
        
    
    def handle_region_selected(self, region):
        """处理选择的区域"""
        if region:
            self.selected_region = region
            self.x = region.x()
            self.y = region.y()
            self.w = region.width()
            self.h = region.height()
            self.btn_crop.setEnabled(True)
            self.region_selection_done = True  # 标记为完成
            # self.region_selection_finished.emit()  # 发射信号，通知选择完成
    
    def crop_selected_region(self):
        """截取选定区域"""
        if self.selected_region and self.full_screenshot:
            # 从全屏截图中截取选定区域
            # *     cropped = self.full_screenshot.copy(self.selected_region)
            cropped = self.screen.grabWindow(0, 
                self.x, self.y, 
                self.w, self.h
            )
            
            # 显示截取的区域
            self.label.setPixmap(cropped.scaled(
                self.label.width(), self.label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            self.full_screenshot = cropped  # 更新当前截图
            self.btn_process.setEnabled(True)
    
    def process_and_return_opencv(self):
        """将截图转换为OpenCV格式并返回"""
        if self.full_screenshot:
            # 转换为OpenCV格式
            cv_image = self.qpixmap_to_cv2(self.full_screenshot)
            
            # 存储处理后的图像
            self.processed_cv_image = cv_image
            
            # 在标签中显示处理状态
            h, w, _ = cv_image.shape
            self.label.setText(f"截图已转换为OpenCV格式\n尺寸: {w}x{h} 像素")
            
            # # 示例：显示处理后的图像（取消注释以测试）
            # cv2.imshow("OpenCV Image", cv_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # 返回处理后的图像（在实际应用中可通过信号或方法获取）
            print("OpenCV格式图像已准备好，可以通过 self.processed_cv_image 获取")
            
            return cv_image
        return None
    
    def qpixmap_to_cv2(self,qpixmap):
        """将QPixmap转换为OpenCV格式（BGR），处理步幅问题"""
        qimage = qpixmap.toImage()
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()  # 获取每行字节数（步幅）
        
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        
        # 创建一维数组
        arr = np.array(ptr)
        
        # 考虑步幅，创建正确的二维数组
        bytes_per_pixel = 3  # RGB888是3字节/像素
        if bytes_per_line == width * bytes_per_pixel:
            # 步幅等于宽度，无需额外处理
            arr = arr.reshape(height, width, bytes_per_pixel)
        else:
            # 步幅大于宽度，需要处理填充
            # 创建带填充的数组
            temp = np.zeros((height, bytes_per_line), dtype=np.uint8)
            # 按行复制数据
            for y in range(height):
                start = y * bytes_per_line
                end = start + width * bytes_per_pixel
                temp[y, :width*bytes_per_pixel] = arr[start:end]
            # 重塑为图像数组
            arr = temp[:, :width*bytes_per_pixel].reshape(height, width, bytes_per_pixel)
        
            # 转换为BGR格式供OpenCV使用
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class RegionSelector(QWidget):
    """区域选择窗口"""
    region_selected = pyqtSignal(QRect)
    
    def __init__(self, screenshot):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        
        # 截图显示
        self.background = screenshot
        
        # 选择区域变量
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_drawing = False
    
    def paintEvent(self, event):
        """绘制界面"""
        painter = QPainter(self)
        
        # 绘制背景截图
        painter.drawPixmap(0, 0, self.background)
        
        if self.is_drawing:
            # 绘制选择矩形
            rect = QRect(self.start_point, self.end_point)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            
            # 绘制半透明遮罩
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawRect(0, 0, self.width(), self.height())
            
            # 清除选择区域的遮罩
            painter.setBrush(Qt.transparent)
            painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        self.start_point = event.pos()
        self.end_point = event.pos()
        self.is_drawing = True
        self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.is_drawing:
            self.end_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.is_drawing = False
        
        # 计算选择区域
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())
        
        # 确保选择区域足够大
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            self.region_selected.emit(QRect(x1, y1, x2 - x1, y2 - y1))
        
        self.close()

class DeepSeekAPI:

    """DeepSeek API调用客户端"""
    
    def __init__(self, api_key=None, api_base=None, timeout=30):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or "https://api.deepseek.com/v1"
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("API密钥未设置，请通过环境变量或构造函数传入")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_text(self, messages, model="deepseek-chat", max_tokens=1024, temperature=0.7):
        """调用DeepSeek文本生成API（用于翻译）"""
        endpoint = f"{self.api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "content": content, "raw_data": data}
        except Exception as e:
            return {"success": False, "error": f"API调用失败: {str(e)}"}

class RegAndTranser:
    def __init__(self, image_path):
        self.image_path = image_path
        # self.mode = Mode.ALL()

    def show(self,image,name = "image" ):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_text_boxes(self,image1):
    
        """使用OpenCV检测图像中的文本框"""
        #image = cv2.imread(self.image_path)
        image = image1

        image_o = image.copy()
        if image is None:
            raise ValueError("无法加载图像，请检查路径是否正确")
        self.show(image,'原始')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        self.show(thresh,'二值')  # 二值化
        kernel = np.ones((5, 5), np.uint8)
        img_t1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        self.show(img_t1,'开操作')  # 开
        # img = cv2.bitwise_not(img)
        img_t1 = img_t1.astype(np.uint8)
        contours, _ = cv2.findContours(img_t1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 找轮廓
        text_boxes = []
        image_ = img_t1.copy()
        cv2.drawContours(image_, contours, -1, (0, 255, 0), 2)
        open2 = cv2.morphologyEx(image_, cv2.MORPH_OPEN, kernel)
        self.show(open2,'开2')

        timg = image_.copy()
        # timg = cv2.bitwise_not(timg)
        timg = cv2.cvtColor(timg, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(open2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(timg, contours, -1, (0, 255, 0), 2)

        self.show(timg,'加粗')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret,thresh = cv2.threshold(image,125,255,cv2.THRESH_BINARY)
        cnts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if (0.2 < aspect_ratio < 1) and (w * h > 0):
                cnts.append(cnt)
        contours = cnts
        merged_contours = []
        i = 0
        while i < len(contours) - 1:
            x, y, w, h = cv2.boundingRect(contours[i])
            x_next, y_next, w_next, h_next = cv2.boundingRect(contours[i + 1])
            print("test1:" + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " | " + str(x_next) + " " + str(y_next) + " " + str(w_next) + " " + str(h_next))
            # 判断是否应该合并
            if abs(y_next - y) < 10 and abs(x_next - (x - w_next)) < 10 and abs(h_next - h) < 10:
                print("ok")
                # 计算合并后的边界框
                x_merged = min(x, x_next)
                y_merged = min(y, y_next)
                w_merged = max(x + w, x_next + w_next) - x_merged
                h_merged = max(y + h, y_next + h_next) - y_merged

                # 检查是否还能与下一个轮廓合并
                merged = True
                j = i + 2
                while j < len(contours):
                    x_j, y_j, w_j, h_j = cv2.boundingRect(contours[j])
                    if abs(y_j - y_merged) < 10 and abs(x_j - (x_merged + w_merged)) < 10 and abs(h_j - h_merged) < 10:
                        x_merged = min(x_merged, x_j)
                        y_merged = min(y_merged, y_j)
                        w_merged = max(x_merged + w_merged, x_j + w_j) - x_merged
                        h_merged = max(y_merged + h_merged, y_j + h_j) - y_merged
                        j += 1
                    else:
                        break

                merged_contours.append((x_merged, y_merged, w_merged, h_merged))
                cv2.rectangle(image, (x_merged, y_merged), (x_merged + w_merged, y_merged + h_merged), (255, 0, 0), 2)
                self.show(image)
                i = j  # 跳过所有已合并的轮廓
            else:
                merged_contours.append((x, y, w, h))
                i += 1

        # 处理最后一个轮廓
        if i < len(contours):
            x, y, w, h = cv2.boundingRect(contours[i])
            merged_contours.append((x, y, w, h))

        # 过滤和绘制合并后的边界框
        for x, y, w, h in merged_contours:
            aspect_ratio = w / float(h)
            if (0.2 < aspect_ratio < 10) and (w * h > 10000):
                text_boxes.append((x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.show(image)

        return image_o, image, text_boxes

    
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        self.show(thresh)  # 二值化
        kernel = np.ones((5, 5), np.uint8)
        img_t1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        self.show(img_t1)  # 开
        # img = cv2.bitwise_not(img)
        img_t1 = img_t1.astype(np.uint8)
        contours, _ = cv2.findContours(img_t1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 找轮廓
        text_boxes = []
        image_ = img_t1.copy()
        cv2.drawContours(image_, contours, -1, (0, 255, 0), 2)
        open2 = cv2.morphologyEx(image_, cv2.MORPH_OPEN, kernel)
        self.show(open2)
        return 0

    def my_detect_text_boxes(self,image1):
        # nnn = image1.copy()

        i = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        self.show(thresh,"thresh")  # 二值化
        kernel = np.ones((5, 5), np.uint8)
        img_t2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        self.show(img_t2,"open")  # 开
        # img = cv2.bitwise_not(img)
        img_t2 = img_t2.astype(np.uint8)
        contours, _ = cv2.findContours(img_t2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 找轮廓
        text_boxes = []
        image_ = img_t2.copy()
        cv2.drawContours(image_, contours, -1, (0, 255, 0), 2)
        open2 = cv2.morphologyEx(image_, cv2.MORPH_OPEN, kernel)
        #show(open2,"open2")
        kernel = np.ones((1,9),np.uint8)
        erode = cv2.erode(open2,kernel, iterations=1)
        self.show(erode,"erode")
        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = []

        # nnn = erode.copy()
        # nnn = cv2.cvtColor(nnn,cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            #print(img.size)
            if(float(w*h/img_t2.size) >0.6):
                cnts.append(cnt)
                x,y,w,h = cv2.boundingRect(cnt)
                #cv2.rectangle(nnn,(x,y),(x+w,y+h),(255,0,0),2)
                #show(nnn)
            else:
                print("too small")
        
        areas = [cv2.contourArea(c) for c in cnts]
        sorted_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=False)
        
        # 根据排序索引重新排列轮廓
        sorted_cnts = [cnts[i] for i in sorted_idx]
        cnts = sorted_cnts
        fl_cnts = []
        if cnts: 
            fl_cnts.append(cnts[0])
            for c1 in cnts:
                for c2 in fl_cnts:

                    x1, y1, w1, h1 = cv2.boundingRect(c2)
                    x2, y2, w2, h2 = cv2.boundingRect(c1)
                    
                    # 计算边界
                    rect1_left = x1
                    rect1_right = x1 + w1
                    rect1_top = y1
                    rect1_bottom = y1 + h1
                    
                    rect2_left = x2
                    rect2_right = x2 + w2
                    rect2_top = y2
                    rect2_bottom = y2 + h2
                    
                    # 判断水平和垂直方向是否同时重叠
                    horizontal_overlap = rect1_left < rect2_right and rect2_left < rect1_right
                    vertical_overlap = rect1_top < rect2_bottom and rect2_top < rect1_bottom
                    
                    if not( horizontal_overlap and vertical_overlap):
                        fl_cnts.append(c1)

        image_with_boxes = image1.copy()
        type = BLANK
        if fl_cnts:
            for cnt in fl_cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                image_with_boxes[y:y+h,x:x+w] = (255,255,255)
        else:
            type = GAUSSIAN
            # x,y,w,h = 0,0,image1.shape[1],image1.shape[0]
            # image_with_boxes[y:y+h,x:x+w] = (255,255,255)
            image_with_boxes = cv2.GaussianBlur(image_with_boxes,(3,3),0)

        return image1 , image_with_boxes, fl_cnts ,type

    def put_text_in_rectangle(self,img_cv2, text, rect, font_path="simhei.ttf",type = BLANK):
        textcolor = (0,0,0)
        if type == GAUSSIAN:
            textcolor = (0,0,0)
        """
        在OpenCV格式的图片的指定矩形区域内写入汉字
        
        参数:
        img_cv2: 输入的OpenCV图像（numpy数组，BGR格式）
        text: 要写入的汉字文本
        rect: 矩形区域 (x, y, width, height)
        font_path: 中文字体文件路径，默认使用系统中的黑体
        
        返回:
        result_img: 处理后的OpenCV图像
        """
        # 将OpenCV格式(BGR)转换为PIL格式(RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 解析矩形参数
        x, y, w, h = rect

        #添加段落的分行
        words_per = int(pow(w/h*len(text),1/2))
        print(words_per)
        text = text.replace('\n','')
        text = ''.join(
            text[i:i+words_per] + '\n' 
            if i + words_per < len(text) 
            else text[i:] 
            for i in range(0, len(text), words_per)
        )

        # 加载字体
        try:
            font = ImageFont.truetype(font_path, size=30)
        except IOError:
            # 字体加载失败时的备选方案
            print(f"警告：无法加载字体 {font_path}，尝试使用系统字体")
            try:
                font = ImageFont.truetype("simhei.ttf", size=30)  # Windows
            except IOError:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttf", size=30)  # Linux
                except IOError:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size=30)  # macOS
                    except IOError:
                        print("警告：未找到中文字体，使用默认字体")
                        font = ImageFont.load_default()
        
        # 动态调整字体大小
        font_size = 30
        while True:
            test_font = ImageFont.truetype(font_path, font_size) if font_path else font
            
            # 根据Pillow版本选择计算文本尺寸的方法
            try:
                # 新版本 (>=10.0.0) 使用textbbox
                bbox = draw.textbbox((0, 0), text, font=test_font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                # 旧版本使用textsize
                text_width, text_height = draw.textsize(text, font=test_font)
            
            if text_width > w * 0.9 or text_height > h * 0.9:
                font_size -= 1
                if font_size < 1:
                    break
            else:
                break
        
        # 使用调整后的字体
        font = ImageFont.truetype(font_path, font_size) if font_path else font
        
        # 计算文字居中位置
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        
        # 写入文字（白色）
        draw.text((text_x, text_y), text, font=font, fill=textcolor)
        if type == GAUSSIAN:
            # 获取文字包围盒
            bbox = draw.textbbox((text_x, text_y), text, font=font)
            
            # 添加边距
            padding = 5
            bg_x = max(bbox[0] - padding, 0)
            bg_y = max(bbox[1] - padding, 0)
            bg_width = min(bbox[2] - bbox[0] + 2*padding, img_pil.width - bg_x)
            bg_height = min(bbox[3] - bbox[1] + 2*padding, img_pil.height - bg_y)
            
            # 绘制背景
            draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=(255, 255, 255))
            draw.text((text_x, text_y), text, font=font, fill=textcolor)

        # 将PIL图像转回OpenCV格式(BGR)
        result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return result_img

    def extract_text_from_boxes(self, image, box):
        """从检测到的文本框中提取文本"""
        extracted_texts = []
        x, y, w, h= box
        box_image = image[y:y + h, x:x + w]
        pil_img = Image.fromarray(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB))
        self.show(box_image,"测试1")
        text = pytesseract.image_to_string(pil_img, lang='jpn_vert')

        if text.strip():
            extracted_texts.append({
                'box': (x, y, w, h),
                'text': text.strip()
            })
        return extracted_texts

    def translate_with_deepseek(self, text, source_lang='ja', target_lang='zh'):
        """使用DeepSeek API的聊天接口进行翻译（通过提示词指定翻译任务）"""
        if not text:
            return None

        # 构建翻译提示词（明确要求从源语言翻译到目标语言）
        system_prompt = {
            "role": "system",
            "content": f"你是一个专业的翻译工具，将{source_lang}翻译为{target_lang}，要求准确、流畅,只告诉用户翻译后的内容,其他话不要说。"
        }
        user_prompt = {
            "role": "user",
            "content": text
        }

        # 初始化API客户端
        api = DeepSeekAPI(api_key=DEEPSEEK_API_KEY)
        # 调用生成接口（使用聊天模型）
        response = api.generate_text(messages=[system_prompt, user_prompt])

        if response["success"]:
            return response["content"].strip()
        else:
            print(f"翻译错误: {response.get('error', '未知错误')}")
            return None
    # def changeMode(self):
    #     if self.mode == Mode.ALL():
    #         self.mode =Mode.AUTO()
    #     if self.mode == Mode.AUTO():
    #         self.mode =Mode.ALL()
    # def Mode(self):
        return self.mode
    def process(self,image1):
        cnts = []
        img_o , img_b, cnts ,type= self.my_detect_text_boxes(image1)
        #show(img_b,"box")
        rect =[]
        if True:
            rect.append( (0,0,img_o.shape[1],img_o.shape[0]))

        elif self.mode ==Mode.AUTO():
            if cnts:
                for c in cnts:
                    rect.append(cv2.boundingRect(c))
            else:
                rect.append( (0,0,img_o.shape[1],img_o.shape[0]))

        

        print(f"检测到 {len(rect)} 个文本框")
        for i in rect:
            x, y, w, h = i
            img = img_o[y:y + h, x:x + w]
            self.show(img,"测试2")
            print(i)
            extracted_texts = self.extract_text_from_boxes(img, i)

            print("检测到的文本:")
            print(extracted_texts)
            if not extracted_texts:
                print("ERROR")
            else:
                for item in extracted_texts:
                    print("翻译中...")
                    # 3. 翻译文本（使用新的API调用方式）
                    original_text = item['text']
                    translated_text = self.translate_with_deepseek(original_text)

                    print(f"\n原始文本(日语):\n{original_text}")
                    print(f"\n翻译文本(中文):\n{translated_text if translated_text else '翻译失败'}")

                if not translated_text:
                    translated_text = "未检测到文本"
                img_t = self.put_text_in_rectangle(img_b,translated_text,i,type = type)
                img_b = img_t.copy()
            
        self.show(img_t,"text")
        return img_t

#子窗口
class ScreenshotWindow(QMainWindow):
    """显示截图的独立窗口"""
    def __init__(self, pixmap, x, y):
        super().__init__()  # 不传递parent参数
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint  # 保持最顶层但不限制在主窗口内
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 存储原始坐标
        self.original_x = x
        self.original_y = y
        
        # 主窗口部件和布局
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 截图显示
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        main_layout.addWidget(self.image_label)
        
        # 关闭按钮（浮动在右上角）
        self.close_btn = QPushButton("×", self)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 50, 50, 200);
                color: white;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 200);
            }
        """)
        self.close_btn.clicked.connect(self.close)
        self.close_btn.hide()  # 初始隐藏
        
        # 设置窗口位置和大小
        self.setGeometry(x, y, pixmap.width(), pixmap.height())
        
        # 设置关闭按钮位置（相对于窗口）
        self.close_btn.move(0, 0)  # 固定在窗口左上角
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
        central_widget.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
    
    def enterEvent(self, event):
        """鼠标进入窗口时显示关闭按钮"""
        self.close_btn.show()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开窗口时隐藏关闭按钮"""
        self.close_btn.hide()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        """鼠标按下事件，用于窗口拖动"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动事件，实现窗口拖动"""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and hasattr(self, 'drag_position'):
            del self.drag_position
            event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    tool = ScreenshotTool()
    tool.show()
    sys.exit(app.exec_())