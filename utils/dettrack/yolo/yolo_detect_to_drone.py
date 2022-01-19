"""
    部署到 Jetson Nano 程序
    (1)yolo 目标检测 -> x,y,w,h
    (2)x,y,w,h -> 串口传输 -> drone
"""

import argparse
import random
import struct
import time
from typing import List
import numpy as np
import cv2
import torch
from utils.dettrack.yolo.models.models import Darknet
from utils.dettrack.yolo.utils.datasets import letterbox
from utils.dettrack.yolo.utils.general import non_max_suppression, scale_coords, load_classes, plot_one_box, xyxy2xywh2
import threading


def gstreamer_pipeline(capture_width=1280,
                       capture_height=720,
                       display_width=1280,
                       display_height=720,
                       framerate=30,
                       flip_method=0):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


class Cammer(object):
    """
        摄像头
    """

    def __init__(self, cammer_type: str = 'usb', img_width: int = 1280, img_height: int = 720, save_flag: bool = False,
                 save_path: str = None) -> None:
        super().__init__()

        # 选择摄像头
        if cammer_type == 'usb':
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
        elif cammer_type == "csi":
            self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        # 当前摄像头捕捉的帧图像
        self.current_frame = None
        self.frame_cout = None

        # 是否保存帧图像
        self.save_flag = save_flag
        self.save_path = save_path

    def read_frame(self) -> np.ndarray:
        """
            读取一帧图像
            return (ret, frame)
        """

        # 临时保存当前帧图像
        _, self.current_frame = self.cap.read()
        return self.current_frame

    def save_frame(self) -> None:
        if self.current_frame is not None:
            cv2.imwrite(self.save_path, self.current_frame)


class Detect(object):
    """
        目标检测
    """

    def __init__(self, img_width: int,
                 img_height: int,
                 img_size: int,
                 conf_thres: float,
                 iou_thres: float,
                 yolo_cfg: str,
                 yolo_weights: str,
                 yolo_names: str) -> None:
        super().__init__()

        # images attribute
        self.WIDTH = img_width
        self.HEIGHT = img_height
        self.IMAGE_SIZE = img_size

        self.c_WIDTH = self.WIDTH / 2  # 视野 x 中心
        self.c_HEIGHT = self.HEIGHT / 2  # 视野 y 中心

        # detect attribute
        self.CONF_THRES = conf_thres
        self.IOU_THRES = iou_thres

        # detect config
        self.yolo_cfg = yolo_cfg
        self.yolo_weights = yolo_weights

        self.names = load_classes(yolo_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # half = device.type != 'cpu'
        self.half = False

        # load detect model
        self.model = self.load_model(self.yolo_cfg, self.yolo_weights, self.IMAGE_SIZE)

    def load_model(self, cfg: str, weights: str, img_size: int) -> Darknet:
        """
            load detect model
        """

        try:
            model = Darknet(cfg, img_size)
            try:
                state_dict = torch.load(weights, map_location=self.device)['model']
                model.load_state_dict(state_dict)
            except:
                state_dict = torch.load(weights, map_location=self.device)
                model.load_state_dict(state_dict)

            model.to(self.device).eval()
            if self.half:
                model.half()  # to FP16

        except Exception as e:
            print(e.with_traceback())

            return None
        return model

    def detection(self, image_source) -> np.ndarray:
        """
            对输入帧图像进行目标检测，返回检测结果 (numpy 数组)
            返回三一个二维数组，每一行代表一个检测结果
            image_source：cv2 frame
            return x, y, x, y, conf, cls

            如果没有目标，则返回 None
        """

        with torch.no_grad():
            try:
                # Padded resize
                img = letterbox(image_source, new_shape=self.IMAGE_SIZE)[0]

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(self.device)

                img = img.half() if self.half else img.float()  # uint8 to fp16/32

                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                predict = self.model(img)[0]
                det = non_max_suppression(predict, self.CONF_THRES, self.IOU_THRES, None, agnostic=False)[0]

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_source.shape).round()

                    return det.cpu().numpy()

            except Exception as e:
                print(e.__traceback__())

            return None


class NanoSerial(object):
    """
        Nano 嵌入式设备串口通信类
        实现功能：Nano 通过串口将目标检测的结果发送给目标（无人机）
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: int = 5) -> None:
        super().__init__()

        if port is None or port == '':
            port = self.get_serial_list()
        else:
            self.port = port

        self.baudrate = baudrate
        self.timeout = timeout

        self.seri = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)

        self.READY = False

    def sendData(self, message: bytes) -> int:
        """
            发送数据
        """
        if self.seri.is_open and self.READY:
            # print(message)
            res = self.seri.write(message)

            return res

        return 0

    def receiveData(self) -> None:
        """
            接收数据
        """
        try:
            self.seri.read()
        except Exception as e:
            print(e.__traceback__())

        while self.READY:
            try:
                s = self.seri.readline()
                if s:
                    time.sleep(1)
                    print(s)

            except Exception as e:
                print(e.__traceback__())

    def get_serial_list(self) -> List:
        """
            获取可用串口列表
        """
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)

        results = []
        for i in range(len(port_list)):
            if 'USB' in port_list[i].device:
                results.append(port_list[i].device)

        return results[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image-source', type=str, default='/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images/d15-images/000242.jpg', help='images source')
    parser.add_argument('--image-source', type=str, default='./000242.jpg', help='images source')
    parser.add_argument('--output', type=str, default='./output', help='output folder')
    parser.add_argument('--yolo-weights', type=str, default='./weights/best_yolov4.pt', help='yolo model.pt path(s)')
    parser.add_argument('--yolo-cfg', type=str, default='./cfg/yolov4-tiny.cfg', help='yolo *.cfg path')
    parser.add_argument('--names', type=str, default='./cfg/visDrone.names',
                        help='*yolo detection object name .cfg path')
    parser.add_argument('--img-width', type=int, default=1280, help='images width')
    parser.add_argument('--img-height', type=int, default=720, help='images height')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--cammer-type', type=str, default='usb', help="csi usb")
    args = parser.parse_args()
    print(args)

    nanoSerial = NanoSerial("/dev/ttyUSB0", 115200, 2)
    nanoSerial.READY = True

    t1 = threading.Thread(target=nanoSerial.receiveData)
    t1.start()  # 启动监听线程

    # load detect model (class)
    with torch.no_grad():
        detect = Detect(args)

        # opencv
        if args.cammer_type == 'usb':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.img_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.img_height)
        elif args.cammer_type == "csi":
            cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        # capture images and detect
        while True:
            _, frame = cap.read()

            results = detect.detection(frame)

            if results is None:
                cv2.imshow(f"{args.cammer_type} Camera", frame)
                if cv2.waitKey(10) == ord("q"):
                    break
                continue

            best_result = results[np.argmax(results[:, 4]), :][0:4]  # 取出 x,y,x,y
            xc = (best_result[0] + best_result[2]) / 2 - detect.c_WIDTH
            yc = (best_result[1] + best_result[3]) / 2 - detect.c_HEIGHT

            # 串口发送坐标数据给无人机
            nanoSerial.sendData(str(xc) + ' ' + str(yc))

            # Write results
            for *xyxy, conf, cls in results:
                plot_one_box(xyxy, frame, label=None, color=detect.colors[int(cls)], line_thickness=2)

            cv2.imshow(f"{args.cammer_type} Camera", frame)
            if cv2.waitKey(10) == ord("q"):
                threadLock = threading.Lock()
                threadLock.acquire()
                nanoSerial.READY = False
                threadLock.release()
                break

        cap.release()
        cv2.destroyAllWindows()
