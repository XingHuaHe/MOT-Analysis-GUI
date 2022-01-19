"""
    将训练集中的每张图像中的二维码抠出来，reshape 成指定大小的图片，
    作为 deep sort 目标跟踪模型的外观特征提取模型的训练数据
"""

import os
import argparse
from typing import Tuple
# import cv2
from PIL import Image

def xcycwh_to_x1y1x2y2(xcycwh: list, shape: Tuple) -> Tuple:
    '''
        将 YOLO 格式文件的坐标 xc,yc,w,h 转化为 left,top,w,h 坐标
    '''
    xc, yc, w_, h_ = map(float, xcycwh[1:])
    
    x1 = shape[1] * xc - 0.5 * shape[1] * w_
    y1 = shape[0] * yc - 0.5 * shape[0] * h_
    x2 = shape[1] * xc + 0.5 * shape[1] * w_
    y2 = shape[0] * yc + 0.5 * shape[0] * h_

    return (x1, y1, x2, y2)

def buckle_object(source: str, target: str) -> None:
    
    folds = os.listdir(source)
    count = 0

    for fold in folds:
        fold_path = os.path.join(source, fold)

        if not os.path.isdir(fold_path):
            continue

        filenames = os.listdir(fold_path)
        for filename in filenames:
            filename_path = os.path.join(fold_path, filename)

            label_file_path = filename_path.replace('images', 'labels2', 1).replace('images', 'labels').replace('.jpg', '.txt')

            # img = cv2.imread(filename_path)
            # shape = img.shape
            img = Image.open(filename_path)
            width, height = img.size

            with open(label_file_path, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    x1, y1, x2, y2 = xcycwh_to_x1y1x2y2(line.strip("\n").split(' '), [height, width])
                    # cropImg = img[int(y1):int(y2), int(x1):int(x2)]
                    cropImg = img.crop((int(x1), int(y1), int(x2), int(y2)))

                    # cv2.imwrite(os.path.join(target, f"{count}.jpg"), cropImg)
                    cropImg.save(os.path.join(target, f"{count}.jpg"))
                    count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images", help="images source directory")
    parser.add_argument('--target', type=str, default="/home/linsi/VisDrone-Experiences/Appreance-Feature/outputs/images-qrcode", help="new images directory")
    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)

    buckle_object(args.source, args.target)