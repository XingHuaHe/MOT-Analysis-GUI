import random
import numpy as np
import cv2
import torch
from models.models import Darknet
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, load_classes, plot_one_box, xyxy2xywh2

device = torch.device('cpu')

WIDTH = 1280
HEIGHT = 960
IMAGE_SIZE = 608

CONF_THRES = 0.001
IOU_THRES = 0.6

half = device.type != 'cpu'

try:
    cfg = './cfg/yolov4-tiny.cfg'
    weights = './weights/best_yolov4.pt'
    names = './cfg/visDrone.names'

    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # load model
    model = Darknet(cfg, IMAGE_SIZE)
    try:
        state_dict = torch.load(weights, map_location=device)['model']
        model.load_state_dict(state_dict)
    except:
        state_dict = torch.load(weights, map_location=device)
        model.load_state_dict(state_dict)
    if half:
        model.half()  # to FP16

    # test
    img0 = cv2.imread('/home/linsi/VisDrone-Experiences/QR-Code/images/00000.jpg')

    # Padded resize
    img = letterbox(img0, new_shape=IMAGE_SIZE)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    predict = model(img)[0]
    predict = non_max_suppression(predict, CONF_THRES, IOU_THRES, None, agnostic=False)

    for i, det in enumerate(predict):
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                plot_one_box(xyxy, img0, label=None, color=colors[int(cls)], line_thickness=2)

        cv2.imshow("show", img0)
        cv2.waitKey()

    # # opencv
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # while True:
    #     _, frame = cap.read()


except Exception as e:
    print(e)

finally:
    # cap.release()
    cv2.destroyAllWindows()
