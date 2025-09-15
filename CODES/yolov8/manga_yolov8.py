import cv2
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import os

def detect_manga_balloons():
    # Load yolov3 model configuration & the weights
    net = cv2.dnn.readNet("yolov3_manga109_v2_5000.weights", "yolov3.cfg")
    
    # Set preferable backend (OpenCV 4.x+)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("使用CUDA加速")
    except:
        print("使用CPU模式")

    # Get all the image paths from the test folder
    images_path = glob.glob(r"E:\pyLearn\codes\Projecion\orangeWindow_7_14_betterWindows\CODES\测试图片\comic\*.jpg") + glob.glob(r"E:\pyLearn\codes\Projecion\orangeWindow_7_14_betterWindows\CODES\测试图片\comic\*.png") + glob.glob(r"E:\pyLearn\codes\Projecion\orangeWindow_7_14_betterWindows\CODES\测试图片\comic\*.jpeg")
    
    if not images_path:
        print("在test文件夹中未找到图像文件！")
        return

    # Get output layer names (兼容不同版本的OpenCV)
    layer_names = net.getLayerNames()
    
    # 处理OpenCV版本差异
    try:
        # OpenCV 4.x
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # OpenCV 3.x
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Create output directory for results
    os.makedirs("output", exist_ok=True)

    # For each image in test folder
    for img_path in images_path:
        print(f"处理图像: {os.path.basename(img_path)}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        original_img = img.copy()
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (512, 512), (0, 0, 0), swapRB=True, crop=False)
        
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.25:  # 置信度阈值
                    # Detection output is normalized (center_x, center_y, width, height)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate (x,y) to get (x,y,w,h) bbox format
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # 确保坐标不超出图像边界
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45)
        
        print(f"检测到 {len(boxes)} 个区域，NMS后保留 {len(indexes) if indexes is not None else 0} 个气泡")
        
        # Draw bounding boxes
        font = cv2.FONT_HERSHEY_PLAIN
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        
        if indexes is not None:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                
                # 选择颜色
                color = colors[class_ids[i] % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # 添加标签和置信度
                label = f"text {confidence:.2f}"
                cv2.putText(img, label, (x, y - 5), font, 1, color, 2)

        # Save result
        output_path = os.path.join("output", f"detected_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img)
        
        # Display result
        plt.figure(figsize=(12, 8))
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img_rgb)
        plt.title(f"检测结果: {os.path.basename(img_path)}\n检测到 {len(indexes) if indexes is not None else 0} 个气泡")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def check_model_files():
    """检查必要的模型文件是否存在"""
    required_files = ["yolov3_manga109_v2_5000.weights", "yolov3.cfg"]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"警告: 未找到文件 {file}")
            
    # 检查test文件夹
    if not os.path.exists("test"):
        print("警告: 未找到test文件夹")
        os.makedirs("test", exist_ok=True)
        print("已创建test文件夹，请将测试图像放入其中")

if __name__ == "__main__":
    # 检查文件
    check_model_files()
    
    # 开始检测
    detect_manga_balloons()
    
    print("处理完成！结果保存在output文件夹中")