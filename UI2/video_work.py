import datetime
from queue import Queue
import PIL.Image as Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_agg import FigureCanvasAgg

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import librosa
import os
import moviepy.editor as mp
import random
import imageio

from UI2.inference_gui import infantDetection, moderl_load, center_crop#,audio_model_load,audio_model_detection

Decode2Play = Queue()

class cvDecode(QThread):
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.video_path = ""  # 视频文件路径
        self.changeFlag = 0  # 判断视频文件路径是否更改
        self.cap = cv2.VideoCapture()

    def run(self):
        while self.threadFlag:
            if self.changeFlag == 1 and self.video_path != "":
                self.changeFlag = 0
                self.cap = cv2.VideoCapture(r"" + self.video_path)

            if self.video_path != "":
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    time.sleep(0.03)  # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

                    # 下面两行代码用来控制循环播放的，如果不需要可以删除
                    if frame is None:
                        self.cap = cv2.VideoCapture(r"" + self.video_path)

                    if ret:
                        Decode2Play.put(frame)  # 解码后的数据放到队列中
                    del frame  # 释放资源
                else:
                    #   控制重连
                    self.cap = cv2.VideoCapture(r"" + self.video_path)
                    time.sleep(0.01)


class play_Work(QObject):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playFlag = 0  # 控制播放/暂停
        self.audio_path = ""#音频存放的路径
        self.video_path = ""#视频存放的路径
        #初始化对象
        self.playLabel = QLabel()
        self.speech_signal = QLabel()
        self.recognition_result = QListWidget()
        self.recognition_audio = QListWidget()
        self.data_statistic_show = QLabel()
        self.xintiao_value = QLabel()
        self.tiwen_value = QLabel()
        self.xuetang_value = QLabel()
        self.huxilv_value = QLabel()
        self.wendu_value = QLabel()
        self.tiye_value = QLabel()
        self._value = QLabel()
        self.shijian_value = QLabel()
        self.xintiao_graph = QGraphicsView()
        self.tiwen_graph = QGraphicsView()
        self.xuetang_graph = QGraphicsView()
        self.huxilv_graph = QGraphicsView()
        self.wendu_graph = QGraphicsView()
        self.pingfen_graph = QGraphicsView()

        #导入模型
        self.infant_detection_model = moderl_load()
        # self.aduio_detection_model = audio_model_load()

        #初始化音频相关参数
        self.clip = []
        self.clip2 = [] #不压缩的视频流
        self.count=0
        self.audio_label=["human","device","cry","background"]
        self.sorted_indexes=[]
        self.flag=True
        self.ret=0
        self.cry_time=0
        self.sum=0
        self.start_time=0

        #   不需要重写run方法
    def play(self):
        last_result = ' '
        this_result = ' '
        self.prob_result = np.array([[0,0,0,0,0]])
        # 转换视频数据为音频数据
        # self.audio_path = self.video_to_audio(self.video_path)
        # #获取音频视频的模型
        model=self.infant_detection_model
        # audio_model=self.aduio_detection_model
        # #获取音频的数据
        # self.audio_data,self.samplate=librosa.load(self.audio_path,sr=None)
        # self.auido_duration=int(librosa.get_duration(self.audio_data,self.samplate))
        n = 0
        cnt = 0

        self.graph_init()

        while self.threadFlag:
            # print("进入",switchflag)
            if not Decode2Play.empty():
                #读取队列中的数据并处理
                frame = Decode2Play.get()
                tmp_ = cv2.resize(frame, (480, 270))
                # tmp_ = cv2.resize(frame, (192, 108))
                tmp_ = cv2.cvtColor(tmp_, cv2.COLOR_BGR2RGB)
                self.clip2.append(tmp_)

                if cnt % 5 == 0:
                    cnt=0
                    tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                    tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                    self.clip.append(tmp)

                cnt += 1

                # if self.count>self.auido_duration*2:
                #     self.count=0
                #     self.sorted_indexes=[]
                #     self.flag = True
                #     self.ret = 0
                #     self.cry_time=0
                #     self.sum=0
                #     self.start_time = 0

                # if cnt%12==0 and self.playFlag ==1:
                    # self.audio_clip = audio_model_detection(audio_model, self.audio_data[self.count * self.samplate // 2:(self.count + 1) * self.samplate // 2]
                    #                                         , self.samplate)
                    # print(self.audio_clip)
                    # self.show_audio(self.audio_clip, self.count)
                    # self.count += 1
                    # imagea=self.draw_audio(self.audio_clip).toqpixmap()
                    # self.speech_signal.setPixmap(imagea)
                    # self.speech_signal.setScaledContents(True)

                if len(self.clip) == 16 and self.playFlag == 1:
                    frame, class_result, prob_result = infantDetection(model, clip=self.clip, frame=frame)
                    self.prob_result = prob_result
                    # print(class_result)
                    c=sum(sum(class_result))
                    class_result = str(class_result)
                    # print(prob_result)
                    print(c)

                    # n += 1
                    # self.png_to_gif(n)
                    # todo cv png转gif
                    if c >= 3:
                        n += 1
                        self.png_to_gif(n)

                    this_result = class_result
                    self.clip = []
                    self.clip2 = []

                    # self.get_targrt()
                    self.add_target()
                    self.show_target()
                    self.draw_graph()


                if self.playFlag == 1:
                    frame = cv2.resize(frame, (860, 484), cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                                  QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                    self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在QLabel上展示
                    if (last_result != this_result):
                        now = datetime.datetime.now()
                        # print(self.show_data(prob_result))
                        self.recognition_result.addItem(now.strftime("%m-%d %H:%M:%S") + '识别结果：' + self.show_data(self.prob_result))
                        # self.data_statistic(this_result)  # 统计结果自增1
                        piximage_data_statistic = self.draw_bar().toqpixmap()
                        self.data_statistic_show.setPixmap(piximage_data_statistic)
                        self.data_statistic_show.setScaledContents(True)
                    last_result = this_result
                # cv2.putText(frame, class_result, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
            # time.sleep(0.001)

    def show_data(self, prob_result):
        probs = []
        res = str
        name_list = ['头部', '左手', '左腿', '右手', '右腿']
        for i in range(len(prob_result)):
            if prob_result[i] > 0.5:
                prob_result[i] = prob_result[i]
                probs.append(name_list[i] + '(' + str("%.2f" % (prob_result[i] * 100)) + '%)')
        if len(probs):
            res = ('+'.join(probs))
        else:
            res = "正常"
        return res

    def show_audio(self,data,count):
        if len(self.sorted_indexes)>=3:
            array1=np.array(self.sorted_indexes)
            while True:
                # print(self.ret,count)
                if self.ret>=count:
                    break;
                # print(is_cry(sorted_indexes,i,length),flag,i)
                if self.is_cry(array1, self.ret) == True:
                    self.ret = self.ret + 1
                    if self.flag == True:
                        self.flag = False
                        self.cry_time+=1
                        # print("第 {} 次哭：开始时间:{}s".format(self.cry_time, (ret-1) * 0.5))
                        now=datetime.datetime.now()
                        self.recognition_audio.addItem(now.strftime("%m-%d %H:%M:%S")+"第"
                                                       +str(self.cry_time)+"次哭：开始时间："+str((self.ret-1)*0.5)+"s")
                        self.start_time=(self.ret-1)*0.5
                        print(self.start_time)
                    # print("start:",i, cry_time)
                elif self.is_cry(array1, self.ret) == False:
                    self.ret = self.ret + 1
                    if self.flag == False:
                        self.flag = True
                        # print(" 结束时间:{}s".format((ret - 1) * 0.5))
                        self.recognition_audio.addItem(" 结束时间：" + str((self.ret - 1) * 0.5) + "s")
                        self.sum=self.sum+(self.ret-1)*0.5-self.start_time
                        print(self.sum,self.start_time)
            if count >=self.auido_duration*2:
                if self.flag==False:
                    self.recognition_audio.addItem(" 结束时间：" + str((self.ret) * 0.5) + "s")
                self.recognition_audio.addItem("哭声总时长："+str(self.sum)+"s")
            self.sorted_indexes.append(np.argsort(data, axis=-1)[-1: -4 - 1: -1])
        else:
            self.sorted_indexes.append(np.argsort(data, axis=-1)[-1: -4 - 1: -1])

    # 判断每帧是否哭
    def is_cry(self,sorted_list, index):
        length=len(sorted_list[:, 0])
        if index == 0:
            index += 1
        if index == length - 1:
            index -= 1
        if index > 0 and index < length - 1:
            # print(self.audio_label[sorted_list[index, 0]])
            if self.audio_label[sorted_list[index, 0]] == "cry":
                if self.audio_label[sorted_list[index - 1, 0]] == "cry" or self.audio_label[sorted_list[index + 1, 0]] == "cry":
                    return True
                else:
                    return False
            else:
                if self.audio_label[sorted_list[index - 1, 0]] == "cry" and self.audio_label[sorted_list[index + 1, 0]] == "cry":
                    return True
                else:
                    return False

    # 显示概率
    def draw_bar(self):
        # print("start")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        name_list = ['头部', '左手', '左腿', '右手', '右腿']
        num_list = self.prob_result
        # num_list = [1, 1, 1, 1, 1]
        fig = plt.figure(figsize=(6.5, 4))

        plt.ylim(0, 1)
        plt.bar(range(len(num_list)), num_list, tick_label=name_list)
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_audio(self,data):
        # print("start")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        labels=["人说话声","设备声","哭声","背景噪声"]
        fig = plt.figure(figsize=(6.5, 4))
        # print("end")
        plt.ylim(0, 1)
        plt.bar(range(len(data)), data, tick_label=labels)

        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    # 转化视频数据为音频数据,并返回音频的存放位置
    def video_to_audio(self, video_path):
        # print("start")
        output_path = os.path.join(
            r"C:\Users\dev1se\Desktop\InfantGUI\audio\output_data\\" + video_path.split("/")[-1][:-4] + ".wav")
        my_clip = mp.VideoFileClip(video_path)
        my_clip.audio.write_audiofile(output_path)
        global switchflag
        switchflag = 1
        # print("end")
        return output_path

    def get_targrt(self):

        xintiao = random.randint(100, 125)
        tiwen = random.randint(300, 356) / 10
        xuetang = random.randint(40, 46) / 10
        huxilv = random.randint(40, 45)
        wendu = random.randint(200, 232) / 10
        tiye = 100
        pingfen = 89
        shijian = 4

        target = [xintiao, tiwen, xuetang, huxilv, wendu, tiye, pingfen, shijian]

        return target

    target_list = np.zeros((8, 22), dtype=float)

    def add_target(self):

        new_target = self.get_targrt()
        for i in range(8):
            for j in range(21):
                self.target_list[i][j] = self.target_list[i][j+1]
                # print(self.target_list[i][j])
        for i in range(8):
            self.target_list[i][21] = new_target[i]

    def show_target(self):

        target = []
        for i in range(8):
            target.append(self.target_list[i][21])

        xintiao = target[0]
        tiwen = target[1]
        xuetang = target[2]
        huxilv = target[3]
        wendu = target[4]
        tiye = target[5]
        pingfen = target[6]
        shijian = target[7]


        self.xintiao_value.setText(str(xintiao) + "t/m")
        self.tiwen_value.setText(str(tiwen) + "℃")
        self.xuetang_value.setText(str(xuetang) + "mm/L")
        self.huxilv_value.setText(str(huxilv) + "t/m")
        self.wendu_value.setText(str(wendu) + "℃")
        self.tiye_value.setText(str(tiye) + "mL")
        self.pingfen_value.setText(str(pingfen))
        self.shijian_value.setText(str(shijian) + "h later")

    def png_to_gif(self, n):
        # img_lst = os.listdir(path)
        # frames = []
        frame = self.clip2
        path = "C:\\Users\\dev1se\\Desktop\\InfantGUI\\gif\\save" + str(n) + ".gif"
        imageio.mimsave(path, frame, 'GIF', fps=16)

    def graph_init(self):
        self.xintiao_graph.scene = QGraphicsScene(self)
        self.xintiao_graph.setScene(self.xintiao_graph.scene)
        self.xintiao_graph.path = QPainterPath()
        self.xintiao_graph.path.moveTo(-218, -80)
        self.xintiao_graph.path.lineTo(QPointF(218, -80))
        self.xintiao_graph.path.moveTo(-218, 80)
        self.xintiao_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xintiao_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.xintiao_graph.scene.addItem(item)
        self.xintiao_graph.show()

        self.tiwen_graph.scene = QGraphicsScene(self)
        self.tiwen_graph.setScene(self.tiwen_graph.scene)
        self.tiwen_graph.path = QPainterPath()
        self.tiwen_graph.path.moveTo(-218, -80)
        self.tiwen_graph.path.lineTo(QPointF(218, -80))
        self.tiwen_graph.path.moveTo(-218, 80)
        self.tiwen_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.tiwen_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.tiwen_graph.scene.addItem(item)
        self.tiwen_graph.show()

        self.xuetang_graph.scene = QGraphicsScene(self)
        self.xuetang_graph.setScene(self.xuetang_graph.scene)
        self.xuetang_graph.path = QPainterPath()
        self.xuetang_graph.path.moveTo(-218, -80)
        self.xuetang_graph.path.lineTo(QPointF(218, -80))
        self.xuetang_graph.path.moveTo(-218, 80)
        self.xuetang_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xuetang_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.xuetang_graph.scene.addItem(item)
        self.xuetang_graph.show()

        self.wendu_graph.scene = QGraphicsScene(self)
        self.wendu_graph.setScene(self.wendu_graph.scene)
        self.wendu_graph.path = QPainterPath()
        self.wendu_graph.path.moveTo(-218, -80)
        self.wendu_graph.path.lineTo(QPointF(218, -80))
        self.wendu_graph.path.moveTo(-218, 80)
        self.wendu_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.wendu_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.wendu_graph.scene.addItem(item)
        self.wendu_graph.show()

        self.huxilv_graph.scene = QGraphicsScene(self)
        self.huxilv_graph.setScene(self.huxilv_graph.scene)
        self.huxilv_graph.path = QPainterPath()
        self.huxilv_graph.path.moveTo(-218, -80)
        self.huxilv_graph.path.lineTo(QPointF(218, -80))
        self.huxilv_graph.path.moveTo(-218, 80)
        self.huxilv_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.huxilv_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.huxilv_graph.scene.addItem(item)
        self.huxilv_graph.show()

        self.pingfen_graph.scene = QGraphicsScene(self)
        self.pingfen_graph.setScene(self.pingfen_graph.scene)
        self.pingfen_graph.path = QPainterPath()
        self.pingfen_graph.path.moveTo(-218, -80)
        self.pingfen_graph.path.lineTo(QPointF(218, -80))
        self.pingfen_graph.path.moveTo(-218, 80)
        self.pingfen_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.pingfen_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.pingfen_graph.scene.addItem(item)
        self.pingfen_graph.show()

    def draw_graph(self):
        self.draw_xintiao_graph()
        self.draw_tiwen_graph()
        self.draw_xuetang_graph()
        self.draw_huxilv_graph()
        self.draw_wendu_graph()
        self.draw_pingfen_graph()

    def draw_xintiao_graph(self):
        xintiao_average = 120
        xintiao_line = 0
        xintiao_sensitivity = 1

        self.xintiao_graph.scene = QGraphicsScene(self)
        self.xintiao_graph.setScene(self.xintiao_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.xintiao_graph.scene.addItem(coordinate_item)
        self.xintiao_graph.show()

        self.xintiao_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[xintiao_line][i] == 0:
                self.target_list[xintiao_line][i] = xintiao_average

        self.xintiao_graph.path.moveTo(-218, (int(-self.target_list[xintiao_line][0])
                                              + xintiao_average)*xintiao_sensitivity)

        for i in range(20):
            self.xintiao_graph.path.lineTo(QPointF(-200 + 20 * i, (int(-self.target_list[xintiao_line][i + 1])
                                                                   + xintiao_average)*xintiao_sensitivity))

        self.xintiao_graph.path.lineTo(QPointF(218, (int(-self.target_list[xintiao_line][21])
                                                     + xintiao_average)*xintiao_sensitivity))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xintiao_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.xintiao_graph.scene.addItem(item)
        self.xintiao_graph.show()

    def draw_tiwen_graph(self):
        tiwen_average = 35
        tiwen_line = 1
        tiwen_sensitivity = 5

        self.tiwen_graph.scene = QGraphicsScene(self)
        self.tiwen_graph.setScene(self.tiwen_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.tiwen_graph.scene.addItem(coordinate_item)
        self.tiwen_graph.show()

        self.tiwen_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[tiwen_line][i] == 0:
                self.target_list[tiwen_line][i] = tiwen_average

        self.tiwen_graph.path.moveTo(-218, (int(-self.target_list[tiwen_line][0])
                                            + tiwen_average)*tiwen_sensitivity)

        for i in range(20):
            self.tiwen_graph.path.lineTo(QPointF(-200 + 20 * i, (int(-self.target_list[tiwen_line][i + 1])
                                                                 + tiwen_average)*tiwen_sensitivity))
        self.tiwen_graph.path.lineTo(QPointF(218, (int(-self.target_list[tiwen_line][21])
                                                   + tiwen_average)*tiwen_sensitivity))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.tiwen_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.tiwen_graph.scene.addItem(item)
        self.tiwen_graph.show()

    def draw_xuetang_graph(self):
        xuetang_average = 4
        xuetang_line = 2
        xuetang_sensitivity = 20

        self.xuetang_graph.scene = QGraphicsScene(self)
        self.xuetang_graph.setScene(self.xuetang_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.xuetang_graph.scene.addItem(coordinate_item)
        self.xuetang_graph.show()

        self.xuetang_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[xuetang_line][i] == 0:
                self.target_list[xuetang_line][i] = xuetang_average

        self.xuetang_graph.path.moveTo(-218, int((-self.target_list[xuetang_line][0]
                                                  + xuetang_average) * xuetang_sensitivity))
        for i in range(20):
            self.xuetang_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[xuetang_line][i+1]
                                                                       + xuetang_average) * xuetang_sensitivity)))
        self.xuetang_graph.path.lineTo(QPointF(218, int((-self.target_list[xuetang_line][21]
                                                         + xuetang_average) * xuetang_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xuetang_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.xuetang_graph.scene.addItem(item)
        self.xuetang_graph.show()

    def draw_huxilv_graph(self):
        huxilv_average = 45
        huxilv_line = 3
        huxilv_sensitivity = 5

        self.huxilv_graph.scene = QGraphicsScene(self)
        self.huxilv_graph.setScene(self.huxilv_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.huxilv_graph.scene.addItem(coordinate_item)
        self.huxilv_graph.show()

        self.huxilv_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[huxilv_line][i] == 0:
                self.target_list[huxilv_line][i] = huxilv_average

        self.huxilv_graph.path.moveTo(-218, int((-self.target_list[huxilv_line][0]
                                                 + huxilv_average) * huxilv_sensitivity))
        for i in range(20):
            self.huxilv_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[huxilv_line][i + 1]
                                                                      + huxilv_average) * huxilv_sensitivity)))
        self.huxilv_graph.path.lineTo(QPointF(218, int((-self.target_list[huxilv_line][21]
                                                        + huxilv_average) * huxilv_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.huxilv_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.huxilv_graph.scene.addItem(item)
        self.huxilv_graph.show()

    def draw_wendu_graph(self):
        wendu_average = 20
        wendu_line = 4
        wendu_sensitivity = 20

        self.wendu_graph.scene = QGraphicsScene(self)
        self.wendu_graph.setScene(self.wendu_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.wendu_graph.scene.addItem(coordinate_item)
        self.wendu_graph.show()

        self.wendu_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[wendu_line][i] == 0:
                self.target_list[wendu_line][i] = wendu_average

        self.wendu_graph.path.moveTo(-218, int((-self.target_list[wendu_line][0]
                                                + wendu_average) * wendu_sensitivity))
        for i in range(20):
            self.wendu_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[wendu_line][i + 1]
                                                                     + wendu_average) * wendu_sensitivity)))
        self.wendu_graph.path.lineTo(QPointF(218, int((-self.target_list[wendu_line][21]
                                                       + wendu_average) * wendu_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.wendu_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.wendu_graph.scene.addItem(item)
        self.wendu_graph.show()

    def draw_pingfen_graph(self):
        pingfen_average = 85
        pingfen_line = 6
        pingfen_sensitivity = 5

        self.pingfen_graph.scene = QGraphicsScene(self)
        self.pingfen_graph.setScene(self.pingfen_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.pingfen_graph.scene.addItem(coordinate_item)
        self.pingfen_graph.show()

        self.pingfen_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[pingfen_line][i] == 0:
                self.target_list[pingfen_line][i] = pingfen_average

        self.pingfen_graph.path.moveTo(-218, int((-self.target_list[pingfen_line][0]
                                                  + pingfen_average) * pingfen_sensitivity))
        for i in range(20):
            self.pingfen_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[pingfen_line][i + 1]
                                                                       + pingfen_average) * pingfen_sensitivity)))
        self.pingfen_graph.path.lineTo(QPointF(218, int((-self.target_list[pingfen_line][21]
                                                         + pingfen_average) * pingfen_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.pingfen_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.pingfen_graph.scene.addItem(item)
        self.pingfen_graph.show()

# app = QApplication(sys.argv)
# view = QGraphicsView()
# scene = QGraphicsScene()
# view.setScene(scene)
#
# # create two painter paths for curves
# path1 = QPainterPath()
# path1.moveTo(10, 10)
# path1.cubicTo(80, 0, 50, 50, 80, 80)
#
# path2 = QPainterPath()
# path2.moveTo(80, 80)
# path2.cubicTo(50, 50, 80, 0, 10, 10)
#
# # create two graphics items for curves
# item1 = scene.addPath(path1)
# item2 = scene.addPath(path2)
#
# # set pen color and width for curves
# item1.setPen(QPen(Qt.red, 3))
# item2.setPen(QPen(Qt.blue, 3))
#
# view.show()
# sys.exit(app.exec_())
