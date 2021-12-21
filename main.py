import sys

from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from lib.share import SI
from Login_Pane import LoginPane
from Register_Pane import RegisterPane
from MainWindow_Pane import MainWindowPane

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    mainWindow = MainWindowPane()

    mainWindow.show()

    sys.exit(app.exec_())
