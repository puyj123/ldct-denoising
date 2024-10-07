import os
import glob
import tifffile
import numpy as np

def tiffs_to_npy(folder_path, output_npy_path):
    # 获取文件夹下所有 TIFF 文件
    tiff_files = sorted(glob.glob(os.path.join(folder_path, '*.tif')))

    # 循环遍历 TIFF 文件并转换为 NumPy 的 .npy 文件
    for tiff_file in tiff_files:
        # 读取 TIFF 图像
        tiff_data = tifffile.imread(tiff_file)

        # 获取文件名（不包含路径和后缀）
        file_name = os.path.splitext(os.path.basename(tiff_file))[0]

        # 保存为 NumPy 的 .npy 文件
        npy_path = os.path.join(output_npy_path, f"{file_name}.npy")
        np.save(npy_path, tiff_data)

folder = './datasets/稀疏重建/ground truth'
out_path = './datasets/train/target'

tiffs_to_npy(folder,out_path)