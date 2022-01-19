from PyQt5.QtWidgets import QApplication
from MainWindow_Pane import MainWindowPane

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    mainWindow = MainWindowPane()

    mainWindow.show()

    sys.exit(app.exec_())
