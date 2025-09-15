
import os
import shutil
import random
from pathlib import Path

def split_train_test(classified_dir, test_ratio=0.2, random_seed=42):
    """
    从分类好的图片文件夹中随机选择部分图片作为测试集
    
    参数:
    classified_dir: 已经分类好的图片根目录（包含1-102个子文件夹）
    test_ratio: 测试集比例，默认20%
    random_seed: 随机种子，保证结果可复现
    """
    # 设置随机种子，确保每次运行结果一致
    random.seed(random_seed)
    
    # 创建训练集和测试集根目录
    train_dir = os.path.join(os.path.dirname(classified_dir), "train_set")
    test_dir = os.path.join(os.path.dirname(classified_dir), "test_set")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有类别文件夹（1-102）
    class_folders = [f for f in os.listdir(classified_dir) 
                    if os.path.isdir(os.path.join(classified_dir, f))]
    
    # 记录总文件数和测试集文件数
    total_files = 0
    total_test_files = 0
    
    # 遍历每个类别文件夹
    for class_folder in class_folders:
        # 源类别文件夹路径
        src_class_path = os.path.join(classified_dir, class_folder)
        
        # 创建训练集和测试集对应的类别文件夹
        train_class_path = os.path.join(train_dir, class_folder)
        test_class_path = os.path.join(test_dir, class_folder)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        # 获取该类别下的所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(src_class_path) 
                      if os.path.isfile(os.path.join(src_class_path, f)) and 
                      Path(f).suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"警告：类别 {class_folder} 中未找到图片文件")
            continue
        
        # 计算测试集数量
        num_test = max(1, int(len(image_files) * test_ratio))  # 至少保留1个测试样本
        num_train = len(image_files) - num_test
        
        # 随机选择测试集文件
        test_files = random.sample(image_files, num_test)
        train_files = [f for f in image_files if f not in test_files]
        
        # 移动文件到训练集
        for file in train_files:
            src = os.path.join(src_class_path, file)
            dst = os.path.join(train_class_path, file)
            shutil.move(src, dst)
        
        # 移动文件到测试集
        for file in test_files:
            src = os.path.join(src_class_path, file)
            dst = os.path.join(test_class_path, file)
            shutil.move(src, dst)
        
        # 打印该类别的分割情况
        print(f"类别 {class_folder}: 总样本 {len(image_files)}, 训练集 {num_train}, 测试集 {num_test}")
        
        # 更新总数
        total_files += len(image_files)
        total_test_files += num_test
    
    # 打印总体分割情况
    print(f"\n分割完成！")
    print(f"总样本数: {total_files}")
    print(f"训练集样本数: {total_files - total_test_files}")
    print(f"测试集样本数: {total_test_files}")
    print(f"训练集路径: {os.path.abspath(train_dir)}")
    print(f"测试集路径: {os.path.abspath(test_dir)}")

# 使用示例
if __name__ == "__main__":
    # 请替换为你之前分类好的图片根目录
    classified_directory = "E:\\download\\102flowers\\train"
    
    # 可以调整测试集比例，比如0.2表示20%作为测试集
    test_ratio = 0.2
    
    try:
        split_train_test(classified_directory, test_ratio)
    except Exception as e:
        print(f"发生错误: {str(e)}")
