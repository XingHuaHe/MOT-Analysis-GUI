"""
    main function
"""

import os
import argparse
from utils.yolo_to_deepsort_format import yolo_to_deepsort


if __name__ == "__main__":
    
    # yolo_to_deepsort_format.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, default="/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images", help="input images directory, contain images")
    parser.add_argument('--npy-output', type=str, default="/home/linsi/VisDrone-Experiences/Appreance-Feature/outputs/deepsort-format", help="output directory")
    parser.add_argument('--det-output', type=str, default="/home/linsi/VisDrone-Experiences/Appreance-Feature/outputs/deepsort-format", help="output directory")
    parser.add_argument('--model-name', type=str, default='autoencoder', help="the name of model for appearance extract")
    parser.add_argument('--weights', type=str, default="./weights/autoencoder/50_ck.pt")
    parser.add_argument('--image-size', type=int, default=56, help="model input images size")
    args = parser.parse_args()

    os.makedirs(args.det_output, exist_ok=True)

    yolo_to_deepsort(args, args.images_path)