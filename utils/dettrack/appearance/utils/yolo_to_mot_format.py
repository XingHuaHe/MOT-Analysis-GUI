"""
    将 yolo 格式文件生成 mot 格式文件
    xc_,yc_,w_,h_ 转为 frame,id,left,top,w,h,conf,x,y,z
"""

import os
import argparse
from typing import Tuple
import cv2


def xcycwh_to_ltwh(xcycwh: list, shape: Tuple) -> Tuple:
    '''
        将 YOLO 格式文件的坐标 xc,yc,w,h 转化为 left,top,w,h 坐标
    '''
    xc, yc, w_, h_ = map(float, xcycwh[1:])
    
    x1 = shape[1] * xc - 0.5 * shape[1] * w_
    y1 = shape[0] * yc - 0.5 * shape[0] * h_
    x2 = shape[1] * xc + 0.5 * shape[1] * w_
    y2 = shape[0] * yc + 0.5 * shape[0] * h_

    return (x1, y1, (x2-x1), (y2-y1))


def yolo_to_mot(source: str, target: str) -> None:
    
    folds = os.listdir(source)

    for fold in folds:
        fold_path = os.path.join(source, fold)
        output_fold_path = os.path.join(target, fold)

        if not os.path.isdir(fold_path):
            continue

        os.makedirs(output_fold_path, exist_ok=True)

        filenames = os.listdir(fold_path)

        filenames = sorted(filenames)

        frame_count = 1
        for filename in filenames:
            filename_path = os.path.join(fold_path, filename)
            output_filename_path = os.path.join(output_fold_path, filename)

            with open(filename_path, 'r') as f:
                lines = f.readlines()

                img_path = filename_path.replace('labels2', 'images').replace('labels', 'images').replace('.txt', '.jpg')
                img = cv2.imread(img_path)
                shape = img.shape

                with open(output_filename_path, 'a') as of:

                    for line in lines:
                        l, t, w, h = xcycwh_to_ltwh(line.strip("\n").split(' '), shape)

                        of.write(f"{frame_count} -1 {l} {t} {w} {h} 1 -1 -1 -1\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/labels2", help="labels source")
    parser.add_argument('--target', type=str, default="/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/labels-mot-format", help="new labels target")
    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)

    yolo_to_mot(args.source, args.target)