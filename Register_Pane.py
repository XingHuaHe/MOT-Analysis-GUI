from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget
from ui.Ui_Register import Ui_Register
from PyQt5.Qt import *


class RegisterPane(QWidget, Ui_Register):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setupUi(self)

        # self.center()

    # def center(self):
    #     screen = QDesktopWidget().screenGeometry()
    #     size = self.geometry()
    #
    #     newLeft = int((screen.width() - size.width()) / 2)
    #     newTop = int((screen.height() - size.height()) / 2)
    #
    #     self.move(newLeft, newTop)
