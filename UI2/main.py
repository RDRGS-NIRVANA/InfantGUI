from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from UI2.Infant import Ui_MainWindow
from UI2.video_work import cvDecode, play_Work
import sys

class MainWindow(QMainWindow, Ui_MainWindow):
    def get_screen_width_and_height(self):
        self.desktop = QApplication.desktop()
        # 获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.screen_height = self.screenRect.height()
        self.screen_width = self.screenRect.width()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.get_screen_width_and_height()
        #   导入Qt的qss样式
        self.qssFileName = "qss/mainWin.qss"
        self.qss = QSSLoad()
        self.qssFile = self.qss.readQssFile(self.qssFileName)
        self.setStyleSheet(self.qssFile)

        self.setWindowIcon(QIcon("./qt_ui/hdu.png"))

        self.quit_system.clicked.connect(QApplication.quit)

        # 按钮事件绑定
        self.btn_input.clicked.connect(self.load_path)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_pause.setEnabled(False)
        self.init_work()

    def init_work(self):

        #线程1读取视频数据
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        # self.decodework.start()
        #线程2读取音频数据
        # self.audio_module = audio_module()
        # self.audio_module.threadFlag = 1
        # self.audio_module.start()
        #线程3视频播放显示结果
        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.label_video
        # self.playwork.recognition_result = self.recognition_result
        # self.playwork.recognition_audio = self.audiolist_show
        # self.playwork.data_statistic_show = self.data_statistic_show
        # self.playwork.speech_signal = self.auidoplot_show
        self.playwork.xintiao_value = self.xintiao_value
        self.playwork.xuetang_value = self.xuetang_value
        self.playwork.huxilv_value = self.huxilv_value
        self.playwork.wendu_value = self.wendu_value
        self.playwork.tiye_value = self.tiye_value
        self.playwork.tiwen_value = self.tiwen_value
        self.playwork.pingfen_value = self.pingfen_value
        self.playwork.shijian_value = self.shijian_value

        self.playwork.xintiao_graph = self.xintiao_graph
        self.playwork.xuetang_graph = self.xuetang_graph
        self.playwork.huxilv_graph = self.huxilv_graph
        self.playwork.wendu_graph = self.wendu_graph
        self.playwork.tiwen_graph = self.tiwen_graph
        self.playwork.pingfen_graph = self.pingfen_graph

        self.play_thread = QThread()  # 创建线程
        self.playwork.moveToThread(self.play_thread)
        self.play_thread.started.connect(self.playwork.play)  # 线程与类方法进行绑定
        # self.play_thread.start()


    #   视频导入功能
    def load_path(self):
        self.btn_pause.setEnabled(True)
        #   设置文件扩展名过滤,注意用双分号间隔
        fileName, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "Excel Files (*.mp4);;Excel Files (*.avi)")

        # self.audio_module.changeFlag = 1
        # self.audio_module.video_path = r"" + fileName
        # self.audio_module.start()

        self.decodework.changeFlag = 1
        self.decodework.video_path = r"" + fileName
        self.playwork.playFlag = 1
        self.decodework.start()

        self.playwork.video_path=r"" + fileName
        self.play_thread.start()
        # self.video_to_audio(self.decodework.video_path)

    #   暂停/播放功能
    def pause_video(self):
        if self.btn_pause.text() == "暂停":
            self.btn_pause.setText("播放")
            self.playwork.playFlag = 0
        else:
            self.btn_pause.setText("暂停")
            self.playwork.playFlag = 1

    def closeEvent(self, event):
        print("关闭线程")
        # Qt需要先退出循环才能关闭线程
        #线程1
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
        # #线程2
        # if self.audio_module.isRunning():
        #     self.audio_module.threadFlag = 0
        #     self.audio_module.quit()
        #线程3
        if self.play_thread.isRunning():
            self.playwork.threadFlag = 0
            self.play_thread.quit()


class QSSLoad:
    @staticmethod
    def readQssFile(qssFileName):
        with open(qssFileName, 'r', encoding='UTF-8') as file:
            return file.read()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
#    main_window.show_targrt()
    sys.exit(app.exec_())
