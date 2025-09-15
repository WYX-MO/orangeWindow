import sys
import os
import time
import platform
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from numba.cuda.cudadrv.driver import AutoFreePointer
from PIL import ImageGrab
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QMainWindow, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QScreen, QImage,
    QCursor
)

image_path = r"E:\pyLearn\imgs\comic\\1 (3).png"
tesseract_path = r"E:\pyLearn\reses\tesseract.exe"
BLANK = 1
GAUSSIAN = 2
pytesseract.pytesseract.tesseract_cmd = tesseract_path

load_dotenv()
DEEPSEEK_API_KEY = "sk-2b814ccebc6e4d90b76ef40d2dd36f83"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
image_origin_translated = []
img_o_t = [None, None]

class ScreenshotTool(QWidget):
    subwindow_x = 0
    subwindow_y = 0
    region_w = 0
    region_h = 0

    def __init__(self):
        super().__init__()
        self.initUI()
        self.screenshot_windows = []
        self.region_selection_done = False
        self.setWindowTitle("Windows区域截图工具")
        self.setGeometry(100, 100, 600, 400)
        self.screen = None
        self.full_screenshot = None
        self.selected_region = None
        self.processed_cv_image = None
        # 用于存储所有窗口的原始图像和窗口对象
        self.all_windows_data = []

    def runtime(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            print(f"函数运行时间为：{end - start}")
            return result
        return wrapper

    @runtime
    def template_match(self, full_img, fragment):
        res = cv2.matchTemplate(full_img, fragment, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.5:
            return None
        return max_loc

    def capture_screen(self):
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        screenshot_cv2 = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        return screenshot_cv2

    def track(self):
        # 隐藏所有窗口
        for window_data in self.all_windows_data:
            window_data[1].hide()
        
        # 捕获当前屏幕
        Allscreen = self.capture_screen()
        
        # 对每个窗口进行匹配
        for window_data in self.all_windows_data:
            image_origin = window_data[0]
            window = window_data[1]
            
            # 模板匹配
            t = self.template_match(Allscreen, image_origin)
            if t is not None:
                x, y = t
                # 更新窗口位置
                window.setGeometry(x, y, window.width(), window.height())
                window.show()
        
        cv2.imshow("Allscreen", Allscreen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def initUI(self):
        layout = QVBoxLayout()
        btn_layout = QVBoxLayout()

        self.btn_1 = QPushButton("==一键处理.beta==")
        self.btn_1.clicked.connect(self.one)
        self.btn_2 = QPushButton("track")
        self.btn_2.clicked.connect(self.track)
    
        btn_layout.addWidget(self.btn_1)
        btn_layout.addWidget(self.btn_2)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def one(self):
        self.region_selection_done = False
        self.capture_fullscreen()
        self.select_region()
        
        while not self.region_selection_done:
            QApplication.processEvents()
        
        self.show()
        self.crop_selected_region()
        cv_img = self.process_and_return_opencv()
        
        # 处理图像并创建窗口
        RAT = RegAndTranser(image_path)
        processed_cv = RAT.process(cv_img)
        processed_cv = cv2.resize(processed_cv, (cv_img.shape[1], cv_img.shape[0]))
        processed_cv = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2GRAY)

        width = self.region_w
        height = self.region_h

        bytes_per_line = 1 * width
        qimage_processed = QImage(processed_cv.data, width, height, 
                                bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap_processed = QPixmap.fromImage(qimage_processed)
        screenshot_window = ScreenshotWindow(pixmap_processed, self.subwindow_x, self.subwindow_y)
        screenshot_window.show()
        
        # 保存原始图像和窗口对象到列表
        window_data = [cv_img.copy(), screenshot_window]
        self.all_windows_data.append(window_data)
        self.screenshot_windows.append(screenshot_window)

    def translate(self, image_path, cv_img):
        RAT = RegAndTranser(image_path)
        processed_cv = RAT.process(cv_img)
        processed_cv = cv2.resize(processed_cv, (cv_img.shape[1], cv_img.shape[0]))
        processed_cv = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2GRAY)

        width = self.region_w
        height = self.region_h

        bytes_per_line = 1 * width
        qimage_processed = QImage(processed_cv.data, width, height, 
                                bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap_processed = QPixmap.fromImage(qimage_processed)
        screenshot_window = ScreenshotWindow(pixmap_processed, self.subwindow_x, self.subwindow_y)
        screenshot_window.show()
        img_o_t[1] = screenshot_window
        image_origin_translated.append(img_o_t)
        self.screenshot_windows.append(screenshot_window)

    def capture_fullscreen(self):
        self.hide()
        QApplication.processEvents()
        self.screen = QApplication.primaryScreen()
        self.full_screenshot = self.screen.grabWindow(0)
    
    def select_region(self):
        if self.full_screenshot is None:
            return
            
        self.selector = RegionSelector(self.full_screenshot)
        self.selector.region_selected.connect(self.handle_region_selected)
        self.selector.showFullScreen()
        
    def handle_region_selected(self, region):
        if region:
            self.selected_region = region
            self.subwindow_x = region.x()
            self.subwindow_y = region.y()
            self.region_w = region.width()
            self.region_h = region.height()
            self.region_selection_done = True
    
    def crop_selected_region(self):
        if self.selected_region and self.full_screenshot:
            cropped = self.screen.grabWindow(0, 
                self.subwindow_x, self.subwindow_y, 
                self.region_w, self.region_h
            )
            self.full_screenshot = cropped
    
    def process_and_return_opencv(self):
        if self.full_screenshot:
            cv_image = self.qpixmap_to_cv2(self.full_screenshot)
            self.processed_cv_image = cv_image
            h, w, _ = cv_image.shape
            return cv_image
        return None
    
    def qpixmap_to_cv2(self, qpixmap):
        qimage = qpixmap.toImage()
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        bytes_per_pixel = 3  
        
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        
        arr = np.array(ptr)
        
        if bytes_per_line == width * bytes_per_pixel:
            arr = arr.reshape(height, width, bytes_per_pixel)
        else:
            temp = np.zeros((height, bytes_per_line), dtype=np.uint8)
            for y in range(height):
                start = y * bytes_per_line
                end = start + width * bytes_per_pixel
                temp[y, :width*bytes_per_pixel] = arr[start:end]
            arr = temp[:, :width*bytes_per_pixel].reshape(height, width, bytes_per_pixel)
        
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

class RegionSelector(QWidget):
    region_selected = pyqtSignal(QRect)
    
    def __init__(self, screenshot):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.background = screenshot
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_drawing = False
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.background)
        
        if self.is_drawing:
            rect = QRect(self.start_point, self.end_point)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawRect(0, 0, self.width(), self.height())
            painter.setBrush(Qt.transparent)
            painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.end_point = event.pos()
        self.is_drawing = True
        self.update()
    
    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        self.is_drawing = False
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            self.region_selected.emit(QRect(x1, y1, x2 - x1, y2 - y1))
        self.close()

class DeepSeekAPI:
    def __init__(self, api_key=None, api_base=None, timeout=30):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or "https://api.deepseek.com/v1"
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("API密钥未设置")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_text(self, messages, model="deepseek-chat", max_tokens=1024, temperature=0.7):
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

    def show(self, image, name="image"):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def my_detect_text_boxes(self, image1):
        i = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        img_t2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        img_t2 = img_t2.astype(np.uint8)
        contours, _ = cv2.findContours(img_t2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        text_boxes = []
        image_ = img_t2.copy()
        cv2.drawContours(image_, contours, -1, (0, 255, 0), 2)
        open2 = cv2.morphologyEx(image_, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((1,9), np.uint8)
        erode = cv2.erode(open2, kernel, iterations=1)
        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if(float(w*h/img_t2.size) > 0.6):
                cnts.append(cnt)
            else:
                print("too small")
        
        areas = [cv2.contourArea(c) for c in cnts]
        sorted_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=False)
        sorted_cnts = [cnts[i] for i in sorted_idx]
        cnts = sorted_cnts
        fl_cnts = []
        if cnts: 
            fl_cnts.append(cnts[0])
            for c1 in cnts:
                for c2 in fl_cnts:
                    x1, y1, w1, h1 = cv2.boundingRect(c2)
                    x2, y2, w2, h2 = cv2.boundingRect(c1)
                    rect1_left = x1
                    rect1_right = x1 + w1
                    rect1_top = y1
                    rect1_bottom = y1 + h1
                    rect2_left = x2
                    rect2_right = x2 + w2
                    rect2_top = y2
                    rect2_bottom = y2 + h2
                    horizontal_overlap = rect1_left < rect2_right and rect2_left < rect1_right
                    vertical_overlap = rect1_top < rect2_bottom and rect2_top < rect1_bottom
                    if not(horizontal_overlap and vertical_overlap):
                        fl_cnts.append(c1)

        image_with_boxes = image1.copy()
        type = BLANK
        if fl_cnts:
            for cnt in fl_cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                image_with_boxes[y:y+h,x:x+w] = (255,255,255)
        else:
            type = GAUSSIAN
            image_with_boxes = cv2.GaussianBlur(image_with_boxes,(3,3),0)

        return image1, image_with_boxes, fl_cnts, type

    def put_text_in_rectangle(self, img_cv2, text, rect, font_path="simhei.ttf", type=BLANK):
        textcolor = (0,0,0)
        if type == GAUSSIAN:
            textcolor = (0,0,0)

        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        x, y, w, h = rect

        words_per = int(pow(w/h*len(text),1/2))
        text = text.replace('\n','')
        text = ''.join(
            text[i:i+words_per] + '\n' 
            if i + words_per < len(text) 
            else text[i:] 
            for i in range(0, len(text), words_per)
        )

        try:
            font = ImageFont.truetype(font_path, size=30)
        except IOError:
            try:
                font = ImageFont.truetype("simhei.ttf", size=30)
            except IOError:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttf", size=30)
                except IOError:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size=30)
                    except IOError:
                        font = ImageFont.load_default()
        
        font_size = 30
        while True:
            test_font = ImageFont.truetype(font_path, font_size) if font_path else font
            try:
                bbox = draw.textbbox((0, 0), text, font=test_font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=test_font)
            
            if text_width > w * 0.9 or text_height > h * 0.9:
                font_size -= 1
                if font_size < 1:
                    break
            else:
                break
        
        font = ImageFont.truetype(font_path, font_size) if font_path else font
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        
        draw.text((text_x, text_y), text, font=font, fill=textcolor)
        if type == GAUSSIAN:
            bbox = draw.textbbox((text_x, text_y), text, font=font)
            padding = 5
            bg_x = max(bbox[0] - padding, 0)
            bg_y = max(bbox[1] - padding, 0)
            bg_width = min(bbox[2] - bbox[0] + 2*padding, img_pil.width - bg_x)
            bg_height = min(bbox[3] - bbox[1] + 2*padding, img_pil.height - bg_y)
            draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=(255, 255, 255))
            draw.text((text_x, text_y), text, font=font, fill=textcolor)

        result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return result_img

    def extract_text_from_boxes(self, image, box):
        extracted_texts = []
        x, y, w, h = box
        box_image = image[y:y + h, x:x + w]
        pil_img = Image.fromarray(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil_img, lang='jpn_vert')

        if text.strip():
            extracted_texts.append({
                'box': (x, y, w, h),
                'text': text.strip()
            })
        return extracted_texts

    def translate_with_deepseek(self, text, source_lang='ja', target_lang='zh'):
        if not text:
            return None

        system_prompt = {
            "role": "system",
            "content": f"你是一个专业的翻译工具，将{source_lang}翻译为{target_lang}，要求准确、流畅,只告诉用户翻译后的内容,其他话不要说。"
        }
        user_prompt = {
            "role": "user",
            "content": text
        }

        api = DeepSeekAPI(api_key=DEEPSEEK_API_KEY)
        response = api.generate_text(messages=[system_prompt, user_prompt])

        if response["success"]:
            return response["content"].strip()
        else:
            print(f"翻译错误: {response.get('error', '未知错误')}")
            return None

    def process(self, image1):
        cnts = []
        img_o, img_b, cnts, type = self.my_detect_text_boxes(image1)
        rect = []
        rect.append((0, 0, img_o.shape[1], img_o.shape[0]))

        print(f"检测到 {len(rect)} 个文本框")
        for i in rect:
            x, y, w, h = i
            img = img_o[y:y + h, x:x + w]
            extracted_texts = self.extract_text_from_boxes(img, i)

            if not extracted_texts:
                print("ERROR")
            else:
                for item in extracted_texts:
                    original_text = item['text']
                    translated_text = self.translate_with_deepseek(original_text)

                if not translated_text:
                    translated_text = "未检测到文本"
                img_t = self.put_text_in_rectangle(img_b, translated_text, i, type=type)
                img_b = img_t.copy()
            
        return img_t

class ScreenshotWindow(QMainWindow):
    def __init__(self, pixmap, x, y):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.x = x
        self.y = y
        
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        main_layout.addWidget(self.image_label)
        
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
        self.close_btn.hide()
        
        self.setGeometry(x, y, pixmap.width(), pixmap.height())
        self.close_btn.move(0, 0)
        
        self.setMouseTracking(True)
        central_widget.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
    
    def enterEvent(self, event):
        self.close_btn.show()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.close_btn.hide()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'drag_position'):
            del self.drag_position
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    tool = ScreenshotTool()
    tool.show()
    sys.exit(app.exec_())