"""
    将 yolo 格式文件（需要图片和标签）转为 deepsort 算法所需要的输入文件，需要以下几样的支持
    （1）训练好的外观提取模型（参考 train_appreance_feature.py）
    （2）给定的目标检测结果（图像帧包含目标的检测目标的 yolo 格式输出）
"""

import os
import argparse
from typing import Tuple
# import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.autoencoder import AutoEncoder


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


def yolo_to_deepsort(args: argparse.ArgumentParser, images_path: str) -> None:
    """
        将 yolo 格式文件转化为 deepsort 需要的 .npy 文件

        images_path：输入文件夹，包含需要处理的 labels 文件
        output：输入目录
    """

    # 加载外观提取模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size,)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.model_name == "autoencoder":
        model = AutoEncoder(trainable=False, img_size=args.image_size, batch_size=1)
    model.to(device)

    try:
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    except Exception as e:
        print(e.__traceback__())



    # 读取图片文件
    folds = os.listdir(images_path)

    for fold in folds:
        fold_path = os.path.join(images_path, fold)

        if not os.path.isdir(fold_path):
            continue
        
        # det.txt 
        os.makedirs(os.path.join(args.det_output, fold, "det"), exist_ok=True)
        out_label_file_path = os.path.join(args.det_output, fold, "det", "det.txt")
        if os.path.exists(out_label_file_path):
            os.remove(out_label_file_path)

        filenames = os.listdir(fold_path)

        filenames = sorted(filenames)

        results = []
        # frame_count = 1
        frame_count = int(filenames[0].split('.')[0])

        for filename in filenames:
            img_path = os.path.join(fold_path, filename)

            label_file_path = img_path.replace('images', 'labels2', 1).replace('images', 'labels').replace('.jpg', '.txt')

            # 读取图片
            img = Image.open(img_path)
            width, height = img.size

            with open(label_file_path, 'r') as f:
                lines = f.readlines()

                with open(out_label_file_path, 'a') as of:

                    for line in lines:
                        x1, y1, x2, y2 = xcycwh_to_x1y1x2y2(line.strip("\n").split(' '), [height, width])
                        cropImg = img.crop((int(x1), int(y1), int(x2), int(y2)))

                        # l, t, w, h = xcycwh_to_ltwh(line.strip("\n").split(' '), [height, width])

                        mot_format = np.array([frame_count, -1, x1, y1, (x2-x1), (y2-y1), 1, -1, -1, -1])

                        img_t = test_transform(cropImg)

                        # forward
                        feature = model(torch.unsqueeze(img_t.to(device), 0))
                        feature = torch.squeeze(feature).data.cpu().numpy()

                        results.append(np.concatenate((mot_format, feature)))

                        
                        of.write(f"{frame_count} -1 {x1} {y1} {x2-x1} {y2-y1} 1 -1 -1 -1\n")
                            
            frame_count += 1

        npy_path = os.path.join(args.npy_output, fold)
        os.makedirs(npy_path, exist_ok=True)
        npy_path = os.path.join(npy_path, f"{fold}.npy")

        results = np.array(results)
        np.save(npy_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default="/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images", help="input images directory")
    parser.add_argument('--npy-output', type=str, default="/home/linsi/VisDrone-Experiences/Appreance-Feature/outputs/deepsort-format", help="output directory")
    parser.add_argument('--det-output', type=str, default="/home/linsi/VisDrone-Experiences/Appreance-Feature/outputs/deepsort-format", help="detection output")
    parser.add_argument('--model-name', type=str, default='autoencoder', help="the name of model for appearance extract")
    parser.add_argument('--weights', type=str, default="./outputs/90_ck.pt")
    parser.add_argument('--image-size', type=int, default=56, help="model input images size")
    args = parser.parse_args()

    os.makedirs(args.npy_outputs, exist_ok=True)

    yolo_to_deepsort(args.images_path)