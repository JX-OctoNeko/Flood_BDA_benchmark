import os
import numpy as np

from PIL import Image
from skimage.io import imread, imsave

file_path = r"C:\Users\student\DLModels\cdlab\CDLab\src\data\xView\Origin\train\masks\midwest-flooding_00000001_post_disaster.png"
np.set_printoptions(threshold=np.inf)

def count_pixels(file_path):
    # 初始化一个长度为256的数组，用于计数每个像素值的出现次数
    pixel_counts = np.zeros(256, dtype=int)

    img = Image.open(file_path)
    img_array = np.array(img)

    # 如果图片是彩色的，转换成灰度图像
    if len(img_array.shape) == 3:
        img_array = img_array.mean(axis=2).astype(int)

    # 计数并更新结果
    pixel_counts += np.bincount(img_array.ravel(), minlength=256)


    return pixel_counts

pixel_counts = count_pixels(file_path)

# 打印每个像素值及其对应的数量
for pixel_value in range(255,-1,-1):
    # print(f"像素值 {pixel_value} 出现的次数为：{pixel_counts[pixel_value]}")
    pass

im2 = (imread(file_path)).astype("uint8")
print(im2)