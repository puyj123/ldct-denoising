import os

import cv2
import numpy as np


# 设置图片文件夹路径和保存路径
input_folder = 'C:/Users/PU/Desktop/image process'
output_folder = 'C:/Users/PU/Desktop/image process/process'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
thickness = 2
line_type = cv2.LINE_AA
# 遍历文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        color = (0, 255, 255)
        center = (242, 350)
        radius = 108
        image = cv2.imread(input_path)
        cv2.circle(image, center, radius, color, thickness, lineType=line_type)
        cv2.imwrite(output_path, image)

