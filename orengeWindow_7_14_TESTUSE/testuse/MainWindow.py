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



#=============================基本全局变量=========================
# 配置路径
image_path = r"E:\pyLearn\imgs\comic\\1 (3).png"
tesseract_path = r"E:\pyLearn\reses\tesseract.exe"#OCR路径
BLANK = 1
GAUSSIAN = 2

# class Mode:

#     ALL = 1
#     AUTO = 2 
#     def ALL(self):
#         return self.ALL
#     def AUTO(self):
#         return self.AUTO



# 设置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 加载环境变量（如果有）
load_dotenv()
DEEPSEEK_API_KEY = "sk-2b814ccebc6e4d90b76ef40d2dd36f83"  # API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1" # API URL

#Deepseek翻译API调用
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

#图片处理和翻译
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
    def changeMode(self):
        if self.mode == Mode.ALL():
            self.mode =Mode.AUTO()
        if self.mode == Mode.AUTO():
            self.mode =Mode.ALL()
    def Mode(self):
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

class mainWindowPosition:
    def __init__(self, x, y):
        self.x = 0
        self.y = 0

#子窗口
class ScreenshotWindow(QMainWindow):
    """显示截图的独立窗口"""
    def __init__(self, pixmap):
        super().__init__()  # 不传递parent参数
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint  # 保持最顶层但不限制在主窗口内
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 主窗口部件和布局
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
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
        self.close_btn.move(10, 10)  # 固定在左上角
        
        # 截图显示
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        main_layout.addWidget(self.image_label)
        
        # 初始位置（屏幕中央）
        screen_geo = QApplication.primaryScreen().availableGeometry()
        self.move(
            mainWindowPosition.x,mainWindowPosition.y
        )
    
        
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

#橙色窗口
class TransparentWindow(QMainWindow):
    def __init__(self):

        super().__init__()
        # 设置窗口保持在最顶层
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.screenshot_windows = []
        # 窗口属性设置
        self.setWindowTitle("透明截图窗口")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 窗口最小尺寸
        self.setMinimumSize(100, 100)
        
        # 边缘调整大小的区域宽度
        self._edge_size = 10
        
        # 鼠标跟踪相关变量
        self._drag_pos = QPoint()
        self._drag_edge = None
        self._is_dragging_window = False
        
        # 创建主中心部件
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部边框 (20像素高)
        self.top_border = QFrame()
        self.top_border.setFixedHeight(50)
        self.top_border.setStyleSheet("background-color: rgba(255, 200, 150, 150);")
        main_layout.addWidget(self.top_border)
        
        # 中间部分 (包含左右边框和中心透明区域)
        middle_layout = QHBoxLayout()
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)
        
        # 左边框 (20像素宽)
        self.left_border = QFrame()
        self.left_border.setFixedWidth(20)
        self.left_border.setStyleSheet("background-color: rgba(255, 200, 150, 150);")
        middle_layout.addWidget(self.left_border)
        
        # 中心透明区域 (可扩展)
        self.center_area = QWidget()
        self.center_area.setStyleSheet("background: transparent;")
        middle_layout.addWidget(self.center_area, 1)
        
        # 右边框 (20像素宽)
        self.right_border = QFrame()
        self.right_border.setFixedWidth(20)
        self.right_border.setStyleSheet("background-color: rgba(255, 200, 150, 150);")
        middle_layout.addWidget(self.right_border)
        
        main_layout.addLayout(middle_layout, 1)
        
        # 底部边框 (50像素高)
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        
        self.bottom_border = QFrame()
        self.bottom_border.setFixedHeight(20)
        self.bottom_border.setStyleSheet("background-color: rgba(255, 200, 150, 150);")
        bottom_layout.addWidget(self.bottom_border)
        
        # 底部按钮区域
        self.button_area = QWidget()
        self.button_area.setFixedHeight(40)
        self.button_area.setStyleSheet("background: transparent;")
        
        self.button_layout = QHBoxLayout(self.button_area)
        self.button_layout.setContentsMargins(20, 0, 20, 5)
        self.button_layout.setSpacing(10)
        
        # 按钮1 - 截图按钮
        self.btn_capture = QPushButton("截图")
        self.btn_capture.setFixedSize(80, 30)
        self.btn_capture.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 150);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 5px;
                color: black;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 180);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 200);
            }
        """)
        self.btn_capture.clicked.connect(self.capture_transparent_area)
        self.button_layout.addWidget(self.btn_capture)
        
        # 按钮2
        self.btn_2 = QPushButton("按钮2")
        self.btn_2.setFixedSize(80, 30)
        self.btn_2.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 150);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 5px;
                color: black;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 180);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 200);
            }
        """)
        self.btn_2.clicked.connect(lambda: self.button_clicked(2))
        self.button_layout.addWidget(self.btn_2)
        
        # 按钮3
        self.btn_3 = QPushButton("按钮3")
        self.btn_3.setFixedSize(80, 30)
        self.btn_3.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 150);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 5px;
                color: black;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 180);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 200);
            }
        """)
        self.btn_3.clicked.connect(lambda: self.button_clicked(3))
        self.button_layout.addWidget(self.btn_3)
        
        # 退出按钮 (宽度较小)
        self.exit_btn = QPushButton("退出")
        self.exit_btn.setFixedSize(60, 30)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 150, 150, 180);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 5px;
                color: black;
            }
            QPushButton:hover {
                background-color: rgba(255, 100, 100, 200);
            }
            QPushButton:pressed {
                background-color: rgba(255, 50, 50, 220);
            }
        """)
        self.exit_btn.clicked.connect(self.close)
        self.button_layout.addWidget(self.exit_btn)
        
        # 添加弹簧使按钮靠右
        self.button_layout.addStretch(1)
        
        bottom_layout.addWidget(self.button_area)
        main_layout.addLayout(bottom_layout)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # 保存原始边框样式
        self.original_border_style = "background-color: rgba(255, 200, 150, 150);"

    def get_transparent_area_coords(self):
        center_rect = self.center_area.geometry()
        print(f"中心区域窗口坐标: {center_rect}")  # 调试输出
        pos = self.center_area.geometry()
        
        global_top_left = self.center_area.mapToGlobal(QPoint(0, 0))
        global_bottom_right = self.center_area.mapToGlobal(
            QPoint(center_rect.width(), center_rect.height()))
        
        #print(f"转换后的屏幕坐标: {global_top_left}, {global_bottom_right}")  # 调试输出
        mainWindowPosition.x,mainWindowPosition.y = global_top_left.x(),global_top_left.y()
        return global_top_left, global_bottom_right

        # def translate(self,screenshot):
         
    def capture_transparent_area(self):
        try:
            top_left, bottom_right = self.get_transparent_area_coords()
            width = bottom_right.x() - top_left.x()
            height = bottom_right.y() - top_left.y()
            
            # 获取截图
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(0, 
                top_left.x(), top_left.y(), 
                width, height
            )
            
            ##==============================在此更改==========================
            
            # 1. 保存原始截图到桌面
            desktop_path = os.path.expanduser("~/Desktop")
            original_path = os.path.join(desktop_path, "original_screenshot.png")
            screenshot.save(original_path, "PNG")
            
            # 2. 将QPixmap转换为QImage
            qimage = screenshot.toImage()
            
            # 3. 将QImage转换为OpenCV格式
            # 转换为Format_RGBA8888格式确保兼容性
            qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # RGBA
            
            # 转换为BGR格式供OpenCV使用
            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            
            # #=====================SAVE==================================
            set = 0#後續添加其他功能的設置參數
            # path = os.path.abspath(__file__)
            # #保存圖片
            # path = path.rsplit('\\', 1)[0]
            # #==
            # path = path+"\\test2.py"
            # if not os.path.exists(path):
            #     raise FileNotFoundError(f"找不到REG: {path}")
            # cmd = [
            #     "python",
            #     path,
            #     str(set)
            # ]
            # result = subprocess.run(
            #     cmd,
            #     capture_output=True,
            #     text=True
            # )
            
            # if result.returncode != 0:
            #     print(f"错误: {result.stderr}")
            # else:
            #     print(result.stdout)
            # print("transDone 1 to 2 !")
            # #==========================================================

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

            # 5. 将处理后的OpenCV图像转回QImage
            bytes_per_line = 1 * width  # 灰度图每个像素1字节
            qimage_processed = QImage(processed_cv.data, width, height, 
                                    bytes_per_line, QImage.Format_Grayscale8)
            
            # 6. 保存处理后的图片
            processed_path = os.path.join(desktop_path, "processed_screenshot.png")
            qimage_processed.save(processed_path, "PNG")
            
            # 7. 将QImage转回QPixmap用于显示
            pixmap_processed = QPixmap.fromImage(qimage_processed)
            #============================END====================================

            # 创建独立窗口（不关联父窗口）
            screenshot_window = ScreenshotWindow(pixmap_processed)
            screenshot_window.show()
            
            # 管理窗口列表
            self.screenshot_windows.append(screenshot_window)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"截图失败555: {str(e)}")

    def button_clicked(self, button_num):
        if button_num == 2:
            # 关闭所有截图窗口
            for window in self.screenshot_windows:
                window.close()
            self.screenshot_windows = []
        elif button_num == 3:
            RegAndTranser.changeMode()
            QMessageBox.information(self, "切换模式", f"切换为{RegAndTranser.Mode()}")

    def set_borders_transparent(self):
        """将窗口边框设置为完全透明"""
        transparent_style = "background-color: rgba(255, 200, 150, 0);"
        self.top_border.setStyleSheet(transparent_style)
        self.left_border.setStyleSheet(transparent_style)
        self.right_border.setStyleSheet(transparent_style)
        self.bottom_border.setStyleSheet(transparent_style)
    
    def restore_borders(self):
        """恢复窗口边框的原始样式"""
        self.top_border.setStyleSheet(self.original_border_style)
        self.left_border.setStyleSheet(self.original_border_style)
        self.right_border.setStyleSheet(self.original_border_style)
        self.bottom_border.setStyleSheet(self.original_border_style)
    

        QMessageBox.information(self, "按钮点击", f"按下按钮{self.button_num}")
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 记录鼠标按下位置
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            
            # 检测是否在边缘调整区域
            rect = self.rect()
            pos = event.pos()
            
            # 左上角区域
            top_left = QRect(0, 0, self._edge_size, self._edge_size)
            # 右上角区域
            top_right = QRect(rect.width() - self._edge_size, 0, self._edge_size, self._edge_size)
            # 左下角区域
            bottom_left = QRect(0, rect.height() - self._edge_size, self._edge_size, self._edge_size)
            # 右下角区域
            bottom_right = QRect(rect.width() - self._edge_size, rect.height() - self._edge_size, 
                                self._edge_size, self._edge_size)
            
            # 上边缘
            top_edge = QRect(self._edge_size, 0, rect.width() - 2 * self._edge_size, self._edge_size)
            # 左边缘
            left_edge = QRect(0, self._edge_size, self._edge_size, rect.height() - 2 * self._edge_size)
            # 右边缘
            right_edge = QRect(rect.width() - self._edge_size, self._edge_size, 
                              self._edge_size, rect.height() - 2 * self._edge_size)
            # 下边缘
            bottom_edge = QRect(self._edge_size, rect.height() - self._edge_size, 
                                rect.width() - 2 * self._edge_size, self._edge_size)
            
            if top_left.contains(pos):
                self._drag_edge = "top-left"
            elif top_right.contains(pos):
                self._drag_edge = "top-right"
            elif bottom_left.contains(pos):
                self._drag_edge = "bottom-left"
            elif bottom_right.contains(pos):
                self._drag_edge = "bottom-right"
            elif top_edge.contains(pos):
                self._drag_edge = "top"
            elif left_edge.contains(pos):
                self._drag_edge = "left"
            elif right_edge.contains(pos):
                self._drag_edge = "right"
            elif bottom_edge.contains(pos):
                self._drag_edge = "bottom"
            else:
                self._drag_edge = None
                
            # 如果不在边缘区域，检查是否在顶部或底部边框内可以拖动窗口
            if self._drag_edge is None:
                # 检查顶部边框
                if self.top_border.rect().contains(self.top_border.mapFromParent(event.pos())):
                    self._is_dragging_window = True
                # 检查底部边框（排除按钮区域）
                elif (self.bottom_border.rect().contains(self.bottom_border.mapFromParent(event.pos())) and
                      not self.button_area.rect().contains(self.button_area.mapFromParent(event.pos()))):
                    self._is_dragging_window = True
        
        event.accept()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if event.buttons() == Qt.LeftButton:
            if self._is_dragging_window:
                # 移动窗口
                self.move(event.globalPos() - self._drag_pos)
            elif self._drag_edge is not None:
                # 调整窗口大小
                rect = self.geometry()
                global_pos = event.globalPos()
                
                if "top" in self._drag_edge:
                    rect.setTop(global_pos.y())
                if "bottom" in self._drag_edge:
                    rect.setBottom(global_pos.y())
                if "left" in self._drag_edge:
                    rect.setLeft(global_pos.x())
                if "right" in self._drag_edge:
                    rect.setRight(global_pos.x())
                
                # 确保窗口不小于最小尺寸
                if rect.width() < self.minimumWidth():
                    if "left" in self._drag_edge:
                        rect.setLeft(rect.right() - self.minimumWidth())
                    else:
                        rect.setRight(rect.left() + self.minimumWidth())
                if rect.height() < self.minimumHeight():
                    if "top" in self._drag_edge:
                        rect.setTop(rect.bottom() - self.minimumHeight())
                    else:
                        rect.setBottom(rect.top() + self.minimumHeight())
                
                self.setGeometry(rect)
        
        # 更新鼠标光标形状
        self.update_cursor_shape(event.pos())
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self._drag_edge = None
        self._is_dragging_window = False
        event.accept()
    
    def update_cursor_shape(self, pos):
        """根据鼠标位置更新光标形状"""
        # 如果正在拖动窗口，保持箭头光标
        if self._is_dragging_window:
            self.setCursor(Qt.ArrowCursor)
            return
            
        rect = self.rect()
        
        # 左上角区域
        top_left = QRect(0, 0, self._edge_size, self._edge_size)
        # 右上角区域
        top_right = QRect(rect.width() - self._edge_size, 0, self._edge_size, self._edge_size)
        # 左下角区域
        bottom_left = QRect(0, rect.height() - self._edge_size, self._edge_size, self._edge_size)
        # 右下角区域
        bottom_right = QRect(rect.width() - self._edge_size, rect.height() - self._edge_size, 
                            self._edge_size, self._edge_size)
        
        # 上边缘
        top_edge = QRect(self._edge_size, 0, rect.width() - 2 * self._edge_size, self._edge_size)
        # 左边缘
        left_edge = QRect(0, self._edge_size, self._edge_size, rect.height() - 2 * self._edge_size)
        # 右边缘
        right_edge = QRect(rect.width() - self._edge_size, self._edge_size, 
                          self._edge_size, rect.height() - 2 * self._edge_size)
        # 下边缘
        bottom_edge = QRect(self._edge_size, rect.height() - self._edge_size, 
                          rect.width() - 2 * self._edge_size, self._edge_size)
        
        if top_left.contains(pos) or bottom_right.contains(pos):
            self.setCursor(Qt.SizeFDiagCursor)
        elif top_right.contains(pos) or bottom_left.contains(pos):
            self.setCursor(Qt.SizeBDiagCursor)
        elif left_edge.contains(pos) or right_edge.contains(pos):
            self.setCursor(Qt.SizeHorCursor)
        elif top_edge.contains(pos) or bottom_edge.contains(pos):
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
#==========================MAIN===============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransparentWindow()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec_())