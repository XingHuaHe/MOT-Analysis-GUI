# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import struct
from PIL import Image

import cv2
import numpy as np
import torch
from utils.dettrack.appearance.appear import Appearance

from utils.dettrack.deepSort.application_util import preprocessing
from utils.dettrack.deepSort.application_util import visualization
from utils.dettrack.deepSort.deep_sort import nn_matching
from utils.dettrack.deepSort.deep_sort.detection import Detection
from utils.dettrack.deepSort.deep_sort.tracker import Tracker
from utils.dettrack.yolo.yolo_detect_to_drone import Cammer, Detect


def gather_sequence_info(img_height: int, img_width: int):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    seq_info = {
        "sequence_name": "Online",
        "image_filenames": None,
        "detections": None,
        "groundtruth": None,
        "image_size": (img_height, img_width),
        "min_frame_idx": 1,
        "max_frame_idx": 5000,
        "feature_dim": 128,
        "update_ms": 1000 / 1
    }

    return seq_info


def create_detections(playCapture,
                      detect: Detect,
                      appearance: Appearance,
                      conf_thres,
                      min_height=0,
                      cammer: Cammer = None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """

    detection_list = []

    # 读取帧图像
    ret, frame = playCapture.read()

    if ret is False:
        return None, 0

    # 对图像进行目标检测
    results = detect.detection(frame)

    if results is None:
        return detection_list, frame

    best_result = results[np.argmax(results[:, 4]), :][0:4]  # 取出 x,y,x,y
    xc = (best_result[0] + best_result[2]) / 2 - detect.c_WIDTH
    yc = (best_result[1] + best_result[3]) / 2 - detect.c_HEIGHT

    for result in results:
        # cv 转化为 Image，裁剪，并获得目标的外观特征
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cropImg = img.crop((int(result[0]), int(result[1]), int(result[2]), int(result[3])))
        feature = appearance.extract_appearance_feature(cropImg)

        # 获得结果
        if result[3] - result[1] < min_height or result[4] < conf_thres:
            continue
        detection_list.append(
            Detection((result[0], result[1], result[2] - result[0], result[3] - result[1]), result[4], feature))

    return detection_list, frame


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return input_string == "True"


def parse_args():
    """ 
        Parse command line arguments.
    """
    # deep sort
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument("--min-confidence", type=float, default=0.1,
                        help="Detection confidence threshold. Disregard all detections that have a confidence lower than this value.", )
    parser.add_argument("--min-detection-height", type=int, default=0,
                        help="Threshold on the detection bounding box height. Detections with height smaller than this value are disregarded")
    parser.add_argument("--nms-max-overlap", type=float, default=0.3,
                        help="Non-maxima suppression threshold: Maximum detection overlap.")
    parser.add_argument("--max-cosine-distance", type=float, default=10,
                        help="Gating threshold for cosine distance metric (object appearance).")
    parser.add_argument("--nn-budget", type=int, default=10,
                        help="Maximum size of the appearance descriptors gallery. If None, no budget is enforced.")

    parser.add_argument("--serial", type=bool, default=True, help="wether is open serial communication")
    parser.add_argument("--display", default=True, type=bool, help="Show intermediate tracking results")
    parser.add_argument("--save", default=False, type=bool, help="Save intermediate tracking results(.txt)")
    parser.add_argument("--save-video", default=True, type=bool, help="Save intermediate tracking results to videos")
    parser.add_argument("--output-file", type=str, default="./deepSort/outputs/hypotheses.txt",
                        help="Path to the tracking output file. This file will contain the tracking results on completion.")

    # yolo
    parser.add_argument('--cammer-type', type=str, default='usb', help="csi usb")
    parser.add_argument('--yolo-weights', type=str, default='./yolo/weights/best_yolov4.pt',
                        help='yolo model.pt path(s)')
    parser.add_argument('--yolo-cfg', type=str, default='./yolo/cfg/yolov4-tiny.cfg', help='yolo *.cfg path')
    parser.add_argument('--yolo-names', type=str, default='./yolo/cfg/visDrone.names',
                        help='*yolo detection object name .cfg path')
    parser.add_argument('--img-width', type=int, default=1280, help='images width')
    parser.add_argument('--img-height', type=int, default=720, help='images height')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')

    # deep sort appearance
    parser.add_argument('--ds-weights', type=str, default="./appearance/weights/autoencoder/50_ck.pt",
                        help="pre-training model")
    parser.add_argument('--model-name', type=str, default="autoencoder", help="selectived model to train")
    parser.add_argument('--fimg-size', type=int, default=56, help="feature image size")

    args = parser.parse_args()

    return args
