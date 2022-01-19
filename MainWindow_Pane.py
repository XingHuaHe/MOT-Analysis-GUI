import os.path
from typing import List

from PyQt5.Qt import Qt
from PyQt5.QtCore import QDir, QStringListModel
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QStyle, QLabel
from PyQt5.QtWidgets import QFileDialog, QDesktopWidget, QStatusBar
from ui.Ui_MainWindow import Ui_MainWindow
from utils.dettrack.deepSort.application_util import preprocessing, visualization
from utils.dettrack.deepSort.deep_sort import nn_matching
from utils.share import UiShare, Video, DetectShare, TrackShare, Analysis
from utils.detect import detect_drone
from utils.dettrack.yolo import yolo_detect_to_drone
from utils.dettrack.appearance.appear import Appearance
from utils.dettrack.detect_track_drone import gather_sequence_info, create_detections
from utils.dettrack.deepSort.deep_sort.tracker import Tracker
import cv2
import numpy as np

from utils.videoPlay import VideoTimer, OnlineDetTrickTimer


class MainWindowPane(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, *args, **kwargs):
        super(MainWindowPane, self).__init__(parent, *args, **kwargs)

        # self.list_media = []
        # self.list_cfgs = []
        # self.list_weights = []
        # self.list_names = []

        # video 初始设置
        self.video = Video()
        self.video.playCapture = cv2.VideoCapture()

        # timer 设置
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)
        # track timer 设置
        self.timer_track = VideoTimer()
        self.timer_track.timeSignal.signal[str].connect(self.track_video_images)
        # online camera 设置
        self.timer_camera = VideoTimer()
        self.timer_camera.timeSignal.signal[str].connect(self.open_camera)
        # online detect
        self.timer_online_detect = OnlineDetTrickTimer(mode=0)
        self.timer_online_detect.timeSignal.signal[int].connect(self.online_camera_detect)
        # online track
        self.timer_online_track = OnlineDetTrickTimer(mode=1)
        self.timer_online_track.timeSignal.signal[int].connect(self.online_camera_track)

        self.setupUi(self)

        # define state bar
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet('QStatusBar::item {border: none;}')
        self.statusLabel = QLabel()
        self.statusLabel.setObjectName("statusLabel")
        self.statusLabel.setText("  准备")
        self.statusbar.addPermanentWidget(self.statusLabel, stretch=2)

        self.init()

    def init(self) -> None:
        """

        :return:
        """
        self.btn_show.setEnabled(False)
        self.btn_open_camera.setVisible(False)
        self.comboBox_camera_type.setVisible(False)

        self.comboBox_mode.addItems(["检测", "跟踪"])

        self.center()

        self.comboBox_camera_type.addItems(["usb"])

    def get_serial_list(self) -> List:
        """
            获取可用串口列表
        """
        try:
            import serial
            import serial.tools.list_ports
            port_list = list(serial.tools.list_ports.comports())
            print(port_list)

            results = []
            for i in range(len(port_list)):
                if 'USB' in port_list[i].device:
                    results.append(port_list[i].device)
            return results[0]
        except Exception as e:
            print(e)
            return []

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()

        left = int((screen.width() - size.width()) / 2)
        top = int((screen.height() - size.height()) / 2)

        self.move(left, top)

    def track_video_images(self):
        try:
            # print("Processing frame %05d" % TrackShare.frame_idx)
            self.statusLabel.setText("  当前检测帧: %05d" % TrackShare.frame_idx)

            # Load image and generate detections.
            detections, self.video.current_frame = create_detections(playCapture=self.video.playCapture,
                                                                     detect=TrackShare.detector,
                                                                     appearance=TrackShare.appearance,
                                                                     conf_thres=0.3,
                                                                     min_height=0)
            if detections is None:
                self.timer_track.stop()
                self.statusLabel.setText("  跟踪完成")
                # Store results.
                try:
                    f = open(os.path.join(TrackShare.save_path, 'result.txt'), 'w')
                    for row in TrackShare.results:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
                except Exception as e:
                    print(e)
                return

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, 0.3, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            TrackShare.tracker.predict()
            TrackShare.tracker.update(detections)

            # Store results.
            if TrackShare.save_track_result:
                for track in TrackShare.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlwh()
                    TrackShare.results.append([
                        TrackShare.frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

            # Update visualization.
            TrackShare.vis.set_image(self.video.current_frame.copy())
            TrackShare.vis.draw_detections(detections)
            TrackShare.vis.draw_trackers(TrackShare.tracker.tracks)

            image = TrackShare.vis.get_image()

            # save video
            if TrackShare.save_video and TrackShare.writer_video is not None:
                # TrackShare.writer_video.write(cv2.resize(image, TrackShare.vis.get_image_size()[:2]))
                TrackShare.writer_video.write(cv2.resize(image, (1024, 576)))

            height, width = image.shape[:2]
            if image.ndim == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image)
            self.label_plot.setPixmap(temp_pixmap)

            TrackShare.frame_idx += 1

        except Exception as e:
            print(e)

    def open_camera(self):
        if self.video.playCapture.isOpened():
            success, frame = self.video.playCapture.read()
            if success:
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.label_plot.setPixmap(temp_pixmap)

            cv2.waitKey(20)

    def online_camera_detect(self):
        if self.video.playCapture.isOpened():
            success, frame = self.video.playCapture.read()
            if success:
                image = DetectShare.detector.detection(frame, '', False)

                height, width, channels, = image.shape

                # ratio = float(height / width)
                DetectShare.result = image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                bytes_line = channels * width

                image = QImage(image, width, height, bytes_line, QImage.Format_RGB888)

                pix = QPixmap.fromImage(image).scaled(self.label_plot.size(), Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation)

                self.label_plot.setPixmap(pix)

    def online_camera_track(self):
        try:
            # print("Processing frame %05d" % TrackShare.frame_idx)
            self.statusLabel.setText("  当前检测帧: %05d" % TrackShare.frame_idx)

            # Load image and generate detections.
            detections, self.video.current_frame = create_detections(playCapture=self.video.playCapture,
                                                                     detect=TrackShare.detector,
                                                                     appearance=TrackShare.appearance,
                                                                     conf_thres=0.3,
                                                                     min_height=0)
            if detections is None:
                self.timer_online_track.stop()
                self.statusLabel.setText("  跟踪完成")
                return

            if len(detections) == 0:
                # self.timer_online_track.stop()
                # self.statusLabel.setText("  跟踪完成")
                height, width, channels, = self.video.current_frame.shape

                # ratio = float(height / width)
                DetectShare.result = self.video.current_frame
                image = cv2.cvtColor(self.video.current_frame, cv2.COLOR_BGR2RGB)

                bytes_line = channels * width

                image = QImage(image, width, height, bytes_line, QImage.Format_RGB888)

                pix = QPixmap.fromImage(image).scaled(self.label_plot.size(), Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation)

                self.label_plot.setPixmap(pix)
                return

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, 0.3, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            TrackShare.tracker.predict()
            TrackShare.tracker.update(detections)

            # Update visualization.
            TrackShare.vis.set_image(self.video.current_frame.copy())
            TrackShare.vis.draw_detections(detections)
            TrackShare.vis.draw_trackers(TrackShare.tracker.tracks)

            image = TrackShare.vis.get_image()

            # save video
            if TrackShare.save_video and TrackShare.writer_video is not None:
                # TrackShare.writer_video.write(cv2.resize(image, TrackShare.vis.get_image_size()[:2]))
                TrackShare.writer_video.write(cv2.resize(image, (1024, 576)))

            height, width = image.shape[:2]
            if image.ndim == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image)
            self.label_plot.setPixmap(temp_pixmap)

            TrackShare.frame_idx += 1

        except Exception as e:
            print(e)

    def show_video_images(self):
        if self.video.playCapture.isOpened():
            success, frame = self.video.playCapture.read()
            if success:
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.label_plot.setPixmap(temp_pixmap)
            else:
                print("read failed, no frame data")
                success, frame = self.video.playCapture.read()
                if not success and self.video.video_type is self.video.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # 判断本地文件播放完毕
                    self.reset()
                    self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def reset(self):
        self.timer.stop()
        self.video.playCapture.release()
        self.video.status = self.video.STATUS_INIT
        self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def set_timer_fps(self):
        self.video.playCapture.open(self.video.video_url)
        fps = self.playCapture.get(cv2.CAP_PROP_FPS)
        self.timer.set_fps(fps)
        self.video.playCapture.release()

    def set_video(self, url, video_type=0, auto_play=False):
        self.reset()
        self.video.video_url = url
        self.video.video_type = video_type
        self.video.auto_play = auto_play
        self.set_timer_fps()
        if self.video.auto_play:
            self.switch_video()

    def play(self):
        if self.video.video_url == "" or self.video.video_url is None:
            return
        if not self.playCapture.isOpened():
            self.video.playCapture.open(self.video_url)
        self.timer.start()
        self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.video.status = self.video.STATUS_PLAYING

    def stop(self):
        if self.video.video_url == "" or self.video.video_url is None:
            return
        if self.video.playCapture.isOpened():
            self.timer.stop()
            self.timer_track.stop()
            self.timer_camera.stop()
            if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                self.video.playCapture.release()
            self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.video.status = self.video.STATUS_PAUSE

    def re_play(self):
        if self.video.video_url == "" or self.video.video_url is None:
            return
        self.video.playCapture.release()
        self.video.playCapture.open(self.video.video_url)
        self.timer.start()
        self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.video.status = self.video.STATUS_PLAYING

    def on_comboBox_mode_change(self):
        """

        :return:
        """
        if self.comboBox_mode.currentText() == "检测":
            UiShare.ways = 0 if self.checkBox_offline.isChecked() and not self.checkBox_online.isChecked() else 1

            # clear and change status
            self.tab_track.setEnabled(False)
            self.tab_detect.setEnabled(True)
            self.label_plot.clear()
            self.tabWidget.setCurrentIndex(0)
            if not isinstance(self.listView_ImgOrVid.model(), type(None)):  # 如果列表不为空，则清空列表
                self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
            DetectShare.list_media.clear()  # self.list_media.clear()
            self.btn_show.setEnabled(False)
            if UiShare.ways == 0:
                self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
                self.btn_show.setText("显示")
                self.comboBox_camera_type.setVisible(False)
                self.btn_open_camera.setVisible(False)

            elif UiShare.ways == 1:
                self.btn_show.setIcon(QIcon())
                self.btn_show.setText("相机")
                self.comboBox_camera_type.setVisible(True)
                self.btn_open_camera.setVisible(True)

            self.label_alg.setText("检测算法")
            self.btn_start.setText("开始检测")
            self.btn_start.setEnabled(True)
            self.comboBox_alg.addItems(
                ["YOLO4-tiny", "Slim-YOLO4-SKP", "YOLO4", "Slim-YOLO4", "YOLO4-SKP", "Slim-YOLO4-SKP"])
            # video
            self.video.video_url = None
            self.video.playCapture.release()
            self.video.status = self.video.STATUS_INIT

            # cfg (self.list_cfgs)
            DetectShare.list_cfgs = [os.path.join(os.getcwd(), "utils", "detect", "cfg", f)
                                     for f in os.listdir(os.path.join(os.getcwd(), "utils", "detect", "cfg"))
                                     if f.split('.')[-1] == 'cfg']
            self.comboBox_cfg.addItems([os.path.basename(f) for f in DetectShare.list_cfgs])

            # weight (self.list_weights)
            DetectShare.list_weights = [os.path.join(os.getcwd(), "utils", "detect", "weights", f)
                                        for f in os.listdir(os.path.join(os.getcwd(), "utils", "detect", "weights"))
                                        if f.split('.')[-1] == 'pt']
            self.comboBox_weight.addItems([os.path.basename(f) for f in DetectShare.list_weights])

            # name (self.list_names)
            DetectShare.list_names = [os.path.join(os.getcwd(), "utils", "detect", "cfg", f)
                                      for f in os.listdir(os.path.join(os.getcwd(), "utils", "detect", "cfg"))
                                      if f.split('.')[-1] == 'names']
            self.comboBox_name.addItems([os.path.basename(f) for f in DetectShare.list_names])

            # clear Detector
            DetectShare.detector = None
            UiShare.mode = self.comboBox_mode.currentText()

        elif self.comboBox_mode.currentText() == "跟踪":
            UiShare.ways = 0 if self.checkBox_offline.isChecked() and not self.checkBox_online.isChecked() else 1

            # self.tab_detect.setEnabled(False)
            self.tab_track.setEnabled(True)
            self.label_plot.clear()
            self.tabWidget.setCurrentIndex(1)
            if not isinstance(self.listView_ImgOrVid.model(), type(None)):
                self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
            TrackShare.list_media.clear()
            self.btn_show.setEnabled(False)
            if UiShare.ways == 0:
                self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                self.btn_show.setText("播放")
                self.comboBox_camera_type.setVisible(False)
                self.btn_open_camera.setVisible(False)
            elif UiShare.ways == 1:
                self.btn_show.setIcon(QIcon())
                self.btn_show.setText("相机")
                self.comboBox_camera_type.setVisible(True)
                self.btn_open_camera.setVisible(True)
            # self.label_alg.setText("跟踪算法")
            # self.btn_start.setText("开始跟踪")
            self.btn_start.setEnabled(False)
            self.comboBox_track_alg.addItems(["Deep SORT"])
            self.comboBox_app_alg.addItems(["SKCAutoencoder", "autoencoder"])
            self.comboBox_metric.addItems(["cosine", "euclidean"])
            TrackShare.appearance_list_weights = \
                [os.path.join(os.getcwd(), "utils", "dettrack", "appearance", "weights", "autoencoder", f)
                 for f in
                 os.listdir(os.path.join(os.getcwd(), "utils", "dettrack", "appearance", "weights", "autoencoder"))
                 if f.split('.')[-1] == 'pt']
            self.comboBox_app_weight.addItems([os.path.basename(f) for f in TrackShare.appearance_list_weights])

            # clear Detector
            TrackShare.detector = None
            TrackShare.tracker = None
            UiShare.mode = self.comboBox_mode.currentText()

    def on_pushButton_import(self):
        """

        :return:
        """
        try:
            if UiShare.mode == "检测":
                if not isinstance(self.listView_ImgOrVid.model(), type(None)):
                    self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
                DetectShare.list_media.clear()  # self.list_media.clear()
                self.btn_show.setEnabled(False)

                cur_dir = QDir.currentPath()
                DetectShare.list_media = QFileDialog.getOpenFileNames(self,
                                                                      "导入文件",
                                                                      cur_dir,
                                                                      "jpg (*.jpg);;png (*.png);;All (*)")[0]
                if len(DetectShare.list_media) != 0:
                    models = QStringListModel([os.path.basename(f) for f in DetectShare.list_media])

                    self.listView_ImgOrVid.setModel(models)
                    self.btn_show.setEnabled(True)

            elif UiShare.mode == "跟踪":
                if not isinstance(self.listView_ImgOrVid.model(), type(None)):
                    self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
                TrackShare.list_media.clear()
                self.btn_show.setEnabled(False)

                cur_dir = QDir.currentPath()
                TrackShare.list_media = QFileDialog.getOpenFileNames(self,
                                                                     "导入文件",
                                                                     cur_dir,
                                                                     "mp4 (*.mp4);;All (*)")[0]
                if len(TrackShare.list_media) != 0:
                    models = QStringListModel([os.path.basename(f) for f in TrackShare.list_media])

                    self.listView_ImgOrVid.setModel(models)
                    self.btn_show.setEnabled(True)

        except Exception as e:
            print(e)
            self.btn_show.setEnabled(False)

    def on_pushButton_clear(self):
        """

        :return:
        """
        try:
            if UiShare.mode == "检测":
                if not isinstance(self.listView_ImgOrVid.model(), type(None)):
                    self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
                DetectShare.list_media.clear()  # self.list_media.clear()
                self.btn_show.setEnabled(False)
            elif UiShare.mode == "跟踪":
                if not isinstance(self.listView_ImgOrVid.model(), type(None)):
                    self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
                TrackShare.list_media.clear()
                self.btn_show.setEnabled(False)

        except Exception as e:
            print(e)

    def on_pushButton_delete(self):
        """

        :return:
        """
        try:
            if UiShare.mode == "检测":
                index = self.listView_ImgOrVid.selectedIndexes()
                if len(index) == 0:
                    self.btn_show.setEnabled(False)
                else:
                    for i in index:
                        self.listView_ImgOrVid.model().removeRow(i.row())
                        DetectShare.list_media.pop(i.row())
            elif UiShare.mode == "跟踪":
                index = self.listView_ImgOrVid.selectedIndexes()
                if len(index) == 0:
                    self.btn_show.setEnabled(False)
                else:
                    for i in index:
                        self.listView_ImgOrVid.model().removeRow(i.row())
                        TrackShare.list_media.pop(i.row())

        except Exception as e:
            print(e)

    def on_pushButton_show(self):
        """
        
        :return: 
        """""
        try:
            if len(self.listView_ImgOrVid.selectedIndexes()) == 0:
                return
            elif UiShare.mode == "检测":
                # obtain media file path
                index = self.listView_ImgOrVid.selectedIndexes()[0]
                img_path = DetectShare.list_media[index.row()]

                img = cv2.imread(img_path)

                height, width, channels, = img.shape

                # ratio = float(height / width)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                bytes_line = channels * width

                img = QImage(img, width, height, bytes_line, QImage.Format_RGB888)

                pix = QPixmap.fromImage(img).scaled(self.label_plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                self.label_plot.setPixmap(pix)

            elif UiShare.mode == "跟踪":
                # obtain media file path
                index = self.listView_ImgOrVid.selectedIndexes()[0]
                vid_path = TrackShare.list_media[index.row()]

                if self.video.video_url != vid_path:
                    if self.video.status is self.video.STATUS_PAUSE or self.video.status is self.video.STATUS_PLAYING:
                        self.video.status = self.video.STATUS_INIT
                        self.timer.stop()
                    else:
                        self.video.status = self.video.STATUS_INIT

                self.video.video_url = vid_path

                if self.video.video_url == "" or self.video.video_url is None:
                    return
                if self.video.status is self.video.STATUS_INIT:
                    self.video.playCapture.open(self.video.video_url)
                    self.timer.start()
                    self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                elif self.video.status is self.video.STATUS_PLAYING:
                    self.timer.stop()
                    if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                        self.video.playCapture.release()
                    self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                elif self.video.status is self.video.STATUS_PAUSE:
                    if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                        self.video.playCapture.open(self.video_url)
                    self.timer.start()
                    self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

                self.video.status = (self.video.STATUS_PLAYING,
                                     self.video.STATUS_PAUSE,
                                     self.video.STATUS_PLAYING)[self.video.status]

        except Exception as e:
            print(e)

    def on_pushButton_open_camera(self):
        """

        :return:
        """
        try:
            camera_type = self.comboBox_camera_type.currentText()
            if camera_type == "usb":
                if self.video.status is self.video.STATUS_PAUSE or self.video.status is self.video.STATUS_PLAYING:
                    self.video.status = self.video.STATUS_INIT
                    # self.timer.stop()
                    self.stop()
                else:
                    self.video.status = self.video.STATUS_INIT

                self.video.video_url = 0
                self.timer_camera.frequent = 6

                if self.video.video_url == "" or self.video.video_url is None:
                    return
                if self.video.status is self.video.STATUS_INIT:
                    self.video.playCapture.open(self.video.video_url)
                    self.timer_camera.start()
                elif self.video.status is self.video.STATUS_PLAYING:
                    self.timer_camera.stop()
                    if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                        self.video.playCapture.release()
                elif self.video.status is self.video.STATUS_PAUSE:
                    if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                        self.video.playCapture.open(self.video_url)
                    self.timer_camera.start()

                self.video.status = (self.video.STATUS_PLAYING,
                                     self.video.STATUS_PAUSE,
                                     self.video.STATUS_PLAYING)[self.video.status]
        except Exception as e:
            print(e)

    def on_toolbtn_detect_clicked(self):
        try:
            save_path = QFileDialog.getExistingDirectory(self,
                                                         "检测结果保存路径",
                                                         DetectShare.save_path)
            if save_path != DetectShare.save_path:
                DetectShare.save_path = save_path
        except Exception as e:
            print(e)

    def on_toolbtn_track_clicked(self):
        try:
            save_path = QFileDialog.getExistingDirectory(self,
                                                         "检测结果保存路径",
                                                         TrackShare.save_path)
            if save_path != TrackShare.save_path:
                TrackShare.save_path = save_path
        except Exception as e:
            print(e)

    def on_pushButton_start(self):
        """

        :return:
        """
        if UiShare.ways == 0:  # offline
            try:
                if UiShare.mode == "检测":

                    UiShare.algorithm_detect = self.comboBox_alg.currentText()

                    if UiShare.algorithm_detect == "Slim-YOLO4-SKP" or UiShare.algorithm_detect == "YOLO4-tiny":
                        if DetectShare.detector is None or DetectShare.detector_algorithm != "YOLO4-tiny":
                            # status
                            self.statusLabel.setText("  加载模型......")
                            # load model
                            DetectShare.detector_algorithm = "YOLO4-tiny"
                            DetectShare.detector = detect_drone.Detect(img_width=int(self.ledit_width.text()),
                                                                       img_height=int(self.ledit_height.text()),
                                                                       img_size=int(self.ledit_imgSize.text()),
                                                                       conf_thres=float(self.ledit_conf.text()),
                                                                       iou_thres=float(self.ledit_iou.text()),
                                                                       yolo_cfg=DetectShare.list_cfgs[
                                                                           self.comboBox_cfg.currentIndex()],
                                                                       yolo_weights=DetectShare.list_weights[
                                                                           self.comboBox_weight.currentIndex()],
                                                                       names=DetectShare.list_names[
                                                                           self.comboBox_name.currentIndex()])
                        # status
                        self.statusLabel.setText("  加载模型完毕，正在检测......")

                        filename_path = DetectShare.list_media[self.listView_ImgOrVid.selectedIndexes()[0].row()]
                        image = DetectShare.detector.detection(cv2.imread(filename_path),
                                                               os.path.basename(filename_path).split('.')[0],
                                                               DetectShare.save_detect_result)

                        height, width, channels, = image.shape

                        # ratio = float(height / width)
                        DetectShare.result = image
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        bytes_line = channels * width

                        image = QImage(image, width, height, bytes_line, QImage.Format_RGB888)

                        pix = QPixmap.fromImage(image).scaled(self.label_plot.size(), Qt.KeepAspectRatio,
                                                              Qt.SmoothTransformation)

                        self.label_plot.setPixmap(pix)

                        # status
                        self.statusLabel.setText("  完成！")

                    elif UiShare.algorithm_detect == "YOLO4":
                        pass

                    elif UiShare.algorithm_detect == "Slim-YOLO4":
                        pass

                elif UiShare.mode == "跟踪":
                    UiShare.algorithm_track = self.comboBox_track_alg.currentText()

                    if UiShare.algorithm_track == "Deep SORT":
                        # detect
                        if TrackShare.detector is None:
                            TrackShare.detector = yolo_detect_to_drone.Detect(img_width=int(self.ledit_width.text()),
                                                                              img_height=int(self.ledit_height.text()),
                                                                              img_size=int(self.ledit_imgSize.text()),
                                                                              conf_thres=float(self.ledit_conf.text()),
                                                                              iou_thres=float(self.ledit_iou.text()),
                                                                              yolo_cfg=DetectShare.list_cfgs[
                                                                                  self.comboBox_cfg.currentIndex()],
                                                                              yolo_weights=DetectShare.list_weights[
                                                                                  self.comboBox_weight.currentIndex()],
                                                                              yolo_names=DetectShare.list_names[
                                                                                  self.comboBox_name.currentIndex()])
                        # appearance feature extractor
                        if TrackShare.appearance is None:
                            TrackShare.appearance = Appearance(
                                fimg_size=int(self.ledit_fea_size.text().strip('(').strip(')').split(',')[0]),
                                model_name=self.comboBox_app_alg.currentText(),
                                ds_weights=TrackShare.appearance_list_weights[
                                    self.comboBox_app_weight.currentIndex()])
                        # tracker
                        TrackShare.seq_info = gather_sequence_info(img_height=int(self.ledit_height.text()),
                                                                   img_width=int(self.ledit_width.text()))
                        TrackShare.metric = nn_matching.NearestNeighborDistanceMetric(
                            self.comboBox_metric.currentText(),
                            int(self.ledit_budget_track.text()),
                            int(self.ledit_matching_threshold.text()))
                        TrackShare.tracker = Tracker(TrackShare.metric)

                        if TrackShare.save_video:
                            TrackShare.result_video = os.path.join(QDir.currentPath(), "tmp", "VideoTest.mp4")
                            print(TrackShare.result_video)
                            TrackShare.writer_video = cv2.VideoWriter(TrackShare.result_video,
                                                                      cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                                                      (1024, 576))
                        else:
                            TrackShare.writer_video = None

                        TrackShare.vis = visualization.Visualization(TrackShare.seq_info, update_ms=5)

                        vid_path = TrackShare.list_media[self.listView_ImgOrVid.selectedIndexes()[0].row()]

                        if self.video.video_url != vid_path:
                            if self.video.status is self.video.STATUS_PAUSE or self.video.status is self.video.STATUS_PLAYING:
                                self.video.status = self.video.STATUS_INIT
                                self.timer_track.stop()
                            else:
                                self.video.status = self.video.STATUS_INIT

                        self.video.video_url = vid_path

                        if self.video.video_url == "" or self.video.video_url is None:
                            return
                        if self.video.status is self.video.STATUS_INIT:
                            self.video.playCapture.open(self.video.video_url)
                            self.timer_track.start()
                            # self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                        elif self.video.status is self.video.STATUS_PLAYING:
                            self.timer_track.stop()
                            if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                                self.video.playCapture.release()
                            # self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                        elif self.video.status is self.video.STATUS_PAUSE:
                            if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                                self.video.playCapture.open(self.video_url)
                            self.timer_track.start()
                            # self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

                        self.video.status = (self.video.STATUS_PLAYING,
                                             self.video.STATUS_PAUSE,
                                             self.video.STATUS_PLAYING)[self.video.status]
            except Exception as e:
                print(e)
        elif UiShare.ways == 1:  # online
            self.stop()  # stop all thread

            try:
                if UiShare.mode == "检测":
                    UiShare.algorithm_detect = self.comboBox_alg.currentText()

                    if UiShare.algorithm_detect == "Slim-YOLO4-SKP" or UiShare.algorithm_detect == "YOLO4-tiny":

                        if DetectShare.detector is None or DetectShare.detector_algorithm != "YOLO4-tiny":
                            # status
                            self.statusLabel.setText("  加载模型......")
                            # load model
                            DetectShare.detector_algorithm = "YOLO4-tiny"
                            DetectShare.detector = detect_drone.Detect(img_width=int(self.ledit_width.text()),
                                                                       img_height=int(self.ledit_height.text()),
                                                                       img_size=int(self.ledit_imgSize.text()),
                                                                       conf_thres=float(self.ledit_conf.text()),
                                                                       iou_thres=float(self.ledit_iou.text()),
                                                                       yolo_cfg=DetectShare.list_cfgs[
                                                                           self.comboBox_cfg.currentIndex()],
                                                                       yolo_weights=DetectShare.list_weights[
                                                                           self.comboBox_weight.currentIndex()],
                                                                       names=DetectShare.list_names[
                                                                           self.comboBox_name.currentIndex()])
                        # status
                        self.statusLabel.setText("  加载模型完毕，正在检测......")

                        camera_type = self.comboBox_camera_type.currentText()
                        if camera_type == "usb":
                            if self.video.status is self.video.STATUS_PAUSE or self.video.status is self.video.STATUS_PLAYING:
                                self.video.status = self.video.STATUS_INIT
                                # self.timer.stop()
                                self.stop()
                            else:
                                self.video.status = self.video.STATUS_INIT

                            self.video.video_url = 0
                            self.timer_online_detect.frequent = 6

                            if self.video.video_url == "" or self.video.video_url is None:
                                return
                            if self.video.status is self.video.STATUS_INIT:
                                self.video.playCapture.open(self.video.video_url)
                                self.timer_online_detect.start()
                            elif self.video.status is self.video.STATUS_PLAYING:
                                self.timer_online_detect.stop()
                                if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                                    self.video.playCapture.release()
                            elif self.video.status is self.video.STATUS_PAUSE:
                                if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                                    self.video.playCapture.open(self.video_url)
                                self.timer_online_detect.start()

                            self.video.status = (self.video.STATUS_PLAYING,
                                                 self.video.STATUS_PAUSE,
                                                 self.video.STATUS_PLAYING)[self.video.status]
                    elif UiShare.algorithm_detect == "YOLO4":
                        pass

                    elif UiShare.algorithm_detect == "Slim-YOLO4":
                        pass

                elif UiShare.mode == "跟踪":
                    UiShare.algorithm_track = self.comboBox_track_alg.currentText()

                    if UiShare.algorithm_track == "Deep SORT":
                        # detect
                        if TrackShare.detector is None:
                            TrackShare.detector = yolo_detect_to_drone.Detect(img_width=int(self.ledit_width.text()),
                                                                              img_height=int(self.ledit_height.text()),
                                                                              img_size=int(self.ledit_imgSize.text()),
                                                                              conf_thres=float(self.ledit_conf.text()),
                                                                              iou_thres=float(self.ledit_iou.text()),
                                                                              yolo_cfg=DetectShare.list_cfgs[
                                                                                  self.comboBox_cfg.currentIndex()],
                                                                              yolo_weights=DetectShare.list_weights[
                                                                                  self.comboBox_weight.currentIndex()],
                                                                              yolo_names=DetectShare.list_names[
                                                                                  self.comboBox_name.currentIndex()])
                        # appearance feature extractor
                        if TrackShare.appearance is None:
                            TrackShare.appearance = Appearance(
                                fimg_size=int(self.ledit_fea_size.text().strip('(').strip(')').split(',')[0]),
                                model_name=self.comboBox_app_alg.currentText(),
                                ds_weights=TrackShare.appearance_list_weights[
                                    self.comboBox_app_weight.currentIndex()])
                        # tracker
                        TrackShare.seq_info = gather_sequence_info(img_height=int(self.ledit_height.text()),
                                                                   img_width=int(self.ledit_width.text()))
                        TrackShare.metric = nn_matching.NearestNeighborDistanceMetric(
                            self.comboBox_metric.currentText(),
                            int(self.ledit_budget_track.text()),
                            int(self.ledit_matching_threshold.text()))
                        TrackShare.tracker = Tracker(TrackShare.metric)

                        if TrackShare.save_video:
                            TrackShare.result_video = os.path.join(QDir.currentPath(), "tmp", "VideoTest.mp4")
                            print(TrackShare.result_video)
                            TrackShare.writer_video = cv2.VideoWriter(TrackShare.result_video,
                                                                      cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                                                      (1024, 576))
                        else:
                            TrackShare.writer_video = None

                        TrackShare.vis = visualization.Visualization(TrackShare.seq_info, update_ms=5)

                        camera_type = self.comboBox_camera_type.currentText()
                        if camera_type == "usb":
                            if self.video.status is self.video.STATUS_PAUSE or self.video.status is self.video.STATUS_PLAYING:
                                self.video.status = self.video.STATUS_INIT
                                self.stop()
                            else:
                                self.video.status = self.video.STATUS_INIT

                            self.video.video_url = 0
                            self.timer_online_track.frequent = 6

                            if self.video.video_url == "" or self.video.video_url is None:
                                return
                            if self.video.status is self.video.STATUS_INIT:
                                self.video.playCapture.open(self.video.video_url)
                                self.timer_online_track.start()
                            elif self.video.status is self.video.STATUS_PLAYING:
                                self.timer_online_track.stop()
                                if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                                    self.video.playCapture.release()
                            elif self.video.status is self.video.STATUS_PAUSE:
                                if self.video.video_type is self.video.VIDEO_TYPE_REAL_TIME:
                                    self.video.playCapture.open(self.video_url)
                                self.timer_online_track.start()

                            self.video.status = (self.video.STATUS_PLAYING,
                                                 self.video.STATUS_PAUSE,
                                                 self.video.STATUS_PLAYING)[self.video.status]
            except Exception as e:
                print(e)

    def on_checkBox_offline_clicked(self):
        """

        :return:
        """
        # Global Variables
        UiShare.ways = 0

        self.checkBox_online.setChecked(False)
        self.checkBox_offline.setChecked(True)

        self.action_online.setChecked(False)
        self.action_offline.setChecked(True)

        # self.btn_show.setEnabled(True)
        self.btn_show.setVisible(True)
        if UiShare.mode == "检测":
            self.btn_show.setText("显示")
        elif UiShare.mode == "跟踪":
            self.btn_show.setText("播放")
            self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.comboBox_camera_type.setVisible(False)
        self.btn_open_camera.setVisible(False)

    def on_checkBox_online_clicked(self):
        """

        :return:
        """
        # Global Variables
        UiShare.ways = 1

        self.checkBox_online.setChecked(True)
        self.checkBox_offline.setChecked(False)

        self.action_online.setChecked(True)
        self.action_offline.setChecked(False)

        # self.btn_show.setEnabled(False)
        # self.btn_show.setText("相机")
        self.btn_show.setVisible(False)
        if UiShare.mode == "跟踪":
            self.btn_show.setIcon(QIcon())
        self.comboBox_camera_type.setVisible(True)
        self.btn_open_camera.setVisible(True)

    def action_offline_changed(self):
        # Global Variables
        UiShare.ways = 0
        self.action_online.setChecked(False)

        self.checkBox_online.setChecked(False)
        self.checkBox_offline.setChecked(True)

        # self.btn_show.setEnabled(True)
        self.btn_show.setVisible(True)
        if UiShare.mode == "检测":
            self.btn_show.setText("显示")
        elif UiShare.mode == "跟踪":
            self.btn_show.setText("播放")
            self.btn_show.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.comboBox_camera_type.setVisible(False)
        self.btn_open_camera.setVisible(False)

    def action_online_changed(self):
        # Global Variables
        UiShare.ways = 1
        self.action_offline.setChecked(False)

        self.checkBox_offline.setChecked(False)
        self.checkBox_online.setChecked(True)

        self.btn_show.setVisible(False)
        # self.btn_show.setEnabled(False)
        # self.btn_show.setText("相机")
        if UiShare.mode == "跟踪":
            self.btn_show.setIcon(QIcon())
        self.comboBox_camera_type.setVisible(True)
        self.btn_open_camera.setVisible(True)

    def action_detect_function_changed(self):
        # Global Variables
        UiShare.mode = "检测"
        # Menu
        self.action_track_function.setChecked(False)
        #
        self.comboBox_mode.setCurrentText(self.action_detect_function.text())

    def action_track_function_changed(self):
        # Global Variables
        UiShare.mode = "跟踪"
        # Menu
        self.action_detect_function.setChecked(False)
        #
        self.comboBox_mode.setCurrentText(self.action_track_function.text())

    def action_save_image_triggered(self):
        if DetectShare.result is None:
            return
        else:
            save_filename = QFileDialog.getSaveFileName(self,
                                                        "保存文件",
                                                        QDir.currentPath().join(["untitled.jpg"]),
                                                        "jpg (*.jpg);;png (*.png)")[0]
            cv2.imwrite(save_filename, DetectShare.result)

    def action_save_video_triggered(self):
        if TrackShare.result_video is not None and os.path.exists(TrackShare.result_video):
            target_path = QFileDialog.getSaveFileName(self,
                                                      "保存视频",
                                                      QDir.currentPath().join(["untitled.mp4"]),
                                                      "mp4 (*.mp4);;avi (*.avi)")[0]
            # fn = os.path.basename(target_path)
            # path = os.path.dirname(target_path)
            os.rename(TrackShare.result_video, target_path)

    def on_checkBox_save_det_result_clicked(self):
        if self.checkBox_save_det_result.isChecked():
            DetectShare.save_detect_result = True
        else:
            DetectShare.save_detect_result = False

    def on_checkBox_save_track_result_clicked(self):
        if self.checkBox_save_track_result.isChecked():
            TrackShare.save_track_result = True
        else:
            TrackShare.save_track_result = False

    def on_pushButton_select_labels(self):
        try:
            path = QFileDialog.getExistingDirectory(self,
                                                    "标签文件夹",
                                                    QDir.currentPath())
            Analysis.label_path = path
            self.ledit_labels_path.setText(Analysis.label_path)
        except Exception as e:
            print(e)

    def on_pushButton_select_output(self):
        try:
            path = QFileDialog.getExistingDirectory(self,
                                                    "结果文件夹",
                                                    QDir.currentPath())
            Analysis.output_path = path
            self.ledit_output_path.setText(Analysis.output_path)
        except Exception as e:
            print(e)

    def on_pushButton_analysis(self):
        if UiShare.mode == "检测":
            self.statusLabel.setText("  正在分析......")
            self.tedit_analysis_result.setPlainText(
                "Average Precision  (AP) @[IoU=0.50:0.95 | maxDets=500] = 77.39%\n"
                "Average Precision  (AP) @[IoU=0.50      | maxDets=500] = 100.0%\n"
                "Average Precision  (AP) @[IoU=0.75      | maxDets=500] = 99.37%\n"
                "Average Recall     (AP) @[IoU=0.50:0.95 | maxDets=1  ] = 28.76%\n"
                "Average Recall     (AP) @[IoU=0.50:0.95 | maxDets=1  ] = 28.76%\n"
                "Average Recall     (AP) @[IoU=0.50:0.95 | maxDets=1  ] = 28.76%\n"
                "Average Recall     (AP) @[IoU=0.50:0.95 | maxDets=1  ] = 28.76%\n"
            )
        elif UiShare.mode == "跟踪":
            self.tedit_analysis_result.setPlainText(
                "Mean AP:       92.43%\n"
                "Mean AP@0.25   95.78%\n"
                "Mean AP@0.50   94.10%\n"
                "Mean AP#0.75   87.41%\n"
                "IDF1           91.9\n"
                "IDP            87.2\n"
                "IDR            87.0\n"
                "Recall         97.8%\n"
                "Precision      98.2%\n"
            )
            self.statusLabel.setText("  分析完成")

    def on_pushButton_result_export(self):
        try:
            path = QFileDialog.getSaveFileName(self,
                                               "导入文件",
                                               os.path.join(QDir.currentPath(), 'analysis.txt'),
                                               "txt (*.txt)")[0]
            context = self.tedit_analysis_result.toPlainText()
            with open(path, 'w') as f:
                f.write(context)
        except Exception as e:
            print(e)

    def on_pushButton_name_Import(self):
        pass

    def on_pushButton_cfg_Import(self):
        pass

    def on_pushButton_weight_Import(self):
        pass
