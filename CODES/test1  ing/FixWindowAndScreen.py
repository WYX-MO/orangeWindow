import cv2
import os
import numpy as np
import time
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# # 构建完整文件路径
# file_path = os.path.join(current_dir, '同文件夹下的文件.txt')

full_img = cv2.imread(os.path.join(current_dir,"./full_w.png"))
fragment = cv2.imread(os.path.join(current_dir,"./target.png"))

def runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"函数运行时间为：{end - start}")
        return result
    return wrapper


@runtime
def template_match(full_img, fragment):
    res = cv2.matchTemplate(full_img, fragment, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # print (max_val)
    if max_val < 0.5:
        return None
    return max_loc  # 返回最佳匹配位置


t = template_match(full_img, fragment)
if t != None:
    x,y = t
    print(x,y)
    full_img[y:y+fragment.shape[0],x:x+fragment.shape[1]] = np.zeros_like(fragment)
cv2.imshow("full_img",full_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

