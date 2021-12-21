from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget
from ui.Ui_Login import Ui_Login


class LoginPane(QWidget, Ui_Login):

    show_register_pane_signal = pyqtSignal()

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)

        # self.setWindowTitle('MOT GUI')

        self.resize(400, 300)

        # self.status = self.statusBar()

        # self.status.showMessage('water')

        self.center()

    def show_register_pane(self):
        self.show_register_pane_signal.emit()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()

        newLeft = int((screen.width() - size.width()) / 2)
        newTop = int((screen.height() - size.height()) / 2)

        self.move(newLeft, newTop)
