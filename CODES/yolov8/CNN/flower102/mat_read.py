import scipy.io as sio
import numpy as np
import os
import shutil
from pathlib import Path

def classify_images(image_dir, labels, output_base_dir='E:\\download\\102flowers'):
    """
    根据标签数组将图片分类到对应文件夹
    
    参数:
    image_dir: 存放原始图片的文件夹路径
    labels: 包含1-102数字的数组，长度与图片数量相同
    output_base_dir: 分类后图片的根目录
    """
    # 获取图片目录中的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(image_dir) if 
                  os.path.isfile(os.path.join(image_dir, f)) and 
                  Path(f).suffix.lower() in image_extensions]
    
    # 确保图片数量与标签数量一致
    if len(image_files) != len(labels):
        raise ValueError(f"图片数量({len(image_files)})与标签数量({len(labels)})不匹配")
    
    # 确保所有标签都在1-102范围内
    for label in labels:
        if not (1 <= label <= 102):
            raise ValueError(f"标签{label}不在1-102范围内")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 创建102个分类文件夹
    for i in range(1, 103):
        class_dir = os.path.join(output_base_dir, str(i))
        os.makedirs(class_dir, exist_ok=True)
    
    # 按标签分类图片
    for i, (image_file, label) in enumerate(zip(image_files, labels)):
        # 源文件路径
        src_path = os.path.join(image_dir, image_file)
        
        # 目标文件夹路径
        dest_dir = os.path.join(output_base_dir, str(label))
        dest_path = os.path.join(dest_dir, image_file)
        
        # 处理可能的文件名冲突
        counter = 1
        while os.path.exists(dest_path):
            # 如果文件已存在，添加后缀避免覆盖
            name, ext = os.path.splitext(image_file)
            dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        # 复制文件
        shutil.copy2(src_path, dest_path)
        
        # 打印进度
        if (i + 1) % 10 == 0 or i + 1 == len(image_files):
            print(f"已处理 {i + 1}/{len(image_files)} 张图片")
    
    print(f"图片分类完成！结果保存在 {os.path.abspath(output_base_dir)}")

# 使用示例
if __name__ == "__main__":
    
    # 读取.mat文件（替换为你的文件路径）
    mat_data = sio.loadmat("E:\\download\\imagelabels.mat")

    # 提取labels并简化形状（去掉外层多余的括号，从(1,N)转为(N,)）
    labels = mat_data["labels"].squeeze()  # squeeze()用于删除维度为1的轴
    print("标签数组形状：", labels.shape)   # 输出如 (1000,)，代表1000个标签
    print("标签取值范围：", np.min(labels), "~", np.max(labels))  # 查看标签的数值范围，辅助判断类别数
    print("前10个标签：", labels)    # 查看前10个标签的具体值
    t =0
    for i in labels:
        print(i ,end=' ')
        t = t+i
    print(t)
    # 请根据实际情况修改以下路径和标签数组
    image_directory = "E:\\download\\102flowers\\jpg"  # 存放图片的文件夹路径
    label_array = labels   # 替换为你的1-102数字数组
    
    try:
        #classify_images(image_directory, label_array)
        pass
    except Exception as e:
        print(f"发生错误: {str(e)}")
