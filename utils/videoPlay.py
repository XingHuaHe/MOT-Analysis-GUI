import time

from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker


class Communicate(QObject):
    signal = pyqtSignal(str)


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


class OnlineCommunicate(QObject):
    signal = pyqtSignal(int)  # 0: detection 1: Tracking


class OnlineDetTrickTimer(QThread):

    def __init__(self, frequent=5, mode=0):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = OnlineCommunicate()
        self.mutex = QMutex()
        self.mode = mode  # 0: detection 1: tracking

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            if self.mode == 0:
                self.timeSignal.signal.emit(0)
            elif self.mode == 1:
                self.timeSignal.signal.emit(1)
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps
