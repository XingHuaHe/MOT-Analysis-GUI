from PyQt5.QtCore import QDir, QStringListModel
from PyQt5.QtWidgets import QMainWindow, QListView
from PyQt5.QtWidgets import QFileDialog
from ui.Ui_MainWindow import Ui_MainWindow


class MainWindowPane(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, *args, **kwargs):
        super(MainWindowPane, self).__init__(parent, *args, **kwargs)

        self.setupUi(self)

        self.init()

        self.list_media = []

    def init(self) -> None:
        """

        :return:
        """
        self.comboBox_model.addItems(["检测", "跟踪"])
        self.pushButton.setEnabled(False)

    def change_mode(self):
        if self.comboBox_model.currentText() == "检测":
            self.label_Alg.setText("检测算法")
            self.btn_start.setText("开始检测")
            self.comboBox_Alg.addItems(["YOLO4", "Slim-YOLO4", "YOLO4-SKP", "Slim-YOLO4-SKP"])

        elif self.comboBox_model.currentText() == "跟踪":
            self.label_Alg.setText("跟踪算法")
            self.btn_start.setText("开始跟踪")

    def on_pushButton_import(self):

        cur_dir = QDir.currentPath()

        self.list_media = QFileDialog.getOpenFileNames(self, "导入文件", cur_dir, "All (*);;jpg (*.jpg);;png (*.png)")[0]

        if len(self.list_media) != 0:
            listModel = QStringListModel(self.list_media)

            self.listView_ImgOrVid.setModel(listModel)

            self.pushButton.setEnabled(True)

    def on_pushButton_clear(self):
        self.listView_ImgOrVid.model().removeRows(0, self.listView_ImgOrVid.model().rowCount())
        self.pushButton.setEnabled(False)

    def on_listView_ImgOrVid_doubleClicked(self):
        pass
