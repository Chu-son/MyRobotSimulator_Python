import socket
import struct
from math import *
import random
import matplotlib.pyplot as plt
import numpy as np
from my_iterative_closest_point import *
import copy
import cv2
from localization import *


class CommMyRobotSimulator:
    def __init__(self, serverPort = 8000, serverAddr = "localhost"):
        self.serverAddr = serverAddr
        self.serverPort = serverPort

        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.connect((serverAddr, serverPort))

        self.urg_offset = [0.6,0.0]

    # 受信した点群データリストをロボット回転中心を原点とするxy座標に変換
    # 想定するデータリスト構造　:　([　radian ... ,　distance ... ])
    def calc_local_coord(self, dataList):
        boundary = int( len(dataList) / 2)
        radian_list = dataList[:boundary]
        distance_list = dataList[boundary:len(dataList)]

        ret_x_list = []
        ret_y_list = []

        for ( rad , dist ) in zip( radian_list , distance_list ):
            ret_x_list.append( dist * cos( rad ) + self.urg_offset[0])
            ret_y_list.append( - dist * sin( rad ) + self.urg_offset[1])
        
        return ret_x_list , ret_y_list

    # シミュレータにコマンドを送る
    def send_command_to_simulator(self, command):
        command += ' \n'
        self.client_sock.send(command.encode('utf-8'))

    # シミュレータにコマンドを送り，受信したbyteデータをfloat型のリストに変換
    def get_data_list(self, command ):
        self.send_command_to_simulator(command)

        response = ()
        while True:
            receivedata = self.client_sock.recv(1024)
            fm = "f" * int( len(receivedata) / struct.calcsize("f"))
            tmp = struct.unpack( fm, receivedata );
            response += tmp
            if -1000.0 in tmp:
                return response[0:-1]

    # 現在地を取得し，メンバを更新
    def get_movement(self):
        return self.get_data_list( "move get")

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        return self.get_data_list("lrf a")

    # 駆動指令を送信
    def send_driving_command(self, direction, speed):
        command = "move send " + direction + " " + str(speed)
        self.get_data_list(command)

    # 衝突しているオブジェクトの数を取得
    def get_hit_count(self):
        return self.get_data_list("hit")[0]


    # お行儀悪いかもだけど実行時に楽だから。。。
    def make_map_loop(self):
        mapping = MappingSimulator()
        mapping.make_map_loop()

    def Localization_loop(self):
        loc = LocalizationSimulator()
        loc.localization_loop()

class MappingSimulator(CommMyRobotSimulator):
    def __init__(self, serverPort = 8000, serverAddr = "localhost"):
        super().__init__(serverPort, serverAddr)

        self.img = np.zeros((1500, 1500, 1), np.uint8)
        self.icp = MyIterativeClosestPoint()

        self.R = np.identity(2)  #回転行列
        self.t = np.zeros([2,1])  #並進ベクトル

        self.icp.error_data = self.urg_offset

        self.init_pos = []
        self.init_dir = []
        self.now_pos = []
        self.now_dir = []
        
        self.pre_pos = []
        self.pre_dir = []

        self.pre_response = None


    # 受信した現在の座標および姿勢から点群を世界座標に変換
    # 初めて受信した位置姿勢が基準となる
    def calc_global_coord(self, dataList):
        data_array = np.array(dataList)

        pos = [now - init for (now,init) in zip(self.now_pos,self.init_pos)]
        dir = [- radians(now - init) for (now,init) in zip(self.now_dir,self.init_dir)]

        R = numpy.array( [[ cos(dir[1]), sin(dir[1])],
                          [-sin(dir[1]), cos(dir[1])]] )
        T = numpy.array( [[ pos[0]] , [-pos[2]] ] )

        data_array = R.dot(data_array) + T

        return data_array

    def calc_odometry_correction_coord(self, dataList):
        data_array = np.array(dataList)

        pos = [now - pre for (pre, now) in zip(self.pre_pos, self.now_pos)]
        dir = [- radians(now - pre) for (pre, now) in zip(self.pre_dir, self.now_dir)]

        R = numpy.array( [[ cos(dir[1]), sin(dir[1])],
                          [-sin(dir[1]), cos(dir[1])]] )
        T = numpy.array( [[ pos[0]] , [-pos[2]] ] )

        data_array = R.dot(data_array) + T

        return data_array, R, T

    #点群データから画像作成
    def plotPoint2Image(self, data):
        coefficient = 100.0 / 5.0
        origin_x = self.img.shape[0] / 5.0
        #origin_y = self.img.shape[1] / 2.0
        origin_y = 1000
    
        for (raw_x, raw_y) in zip(data[0],data[1]):
            if raw_x == 0 and raw_y == 0:continue
            x = int(raw_x * coefficient + origin_x)
            y = int(raw_y * coefficient + origin_y)
        
            if self.img[y,x] != 250:
               self.img[y,x] += 50
        return self.img

    # 現在地を取得し，メンバを更新
    def get_movement(self):
        response = super().get_movement()
        if self.init_pos == [] and self.init_dir == []:
            self.init_pos = response[0:3]
            self.init_dir = response[3:6]

            self.now_pos = response[0:3]
            self.now_dir = response[3:6]

        self.pre_pos = self.now_pos
        self.pre_dir = self.now_dir

        self.now_pos = response[0:3]
        self.now_dir = response[3:6]

        #rand_range = 0.5
        #rand_noise = random.random() * rand_range * 2 - rand_range
        #self.now_pos = [x + rand_noise for x in response[0:3]]
        #self.now_dir = [x + rand_noise for x in response[3:6]]

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        response = super().get_lrf_data()
        if self.pre_response == None:
            self.pre_response = self.calc_local_coord(response)
            return
        #self.img = self.plotPoint2Image( self.calc_global_coord( self.calc_local_coord(response)))

        #plt.scatter(self.pre_response[0], self.pre_response[1], marker = "o",color = "r",s = 60, label = "data1")
        res = self.calc_local_coord(response)
        new_data, Ro, To = self.calc_odometry_correction_coord(res)
        #plt.scatter(new_data[0,:],new_data[1,:],marker = "o",color = "g",s = 40, label = "data2")
        R1, t1, _ = self.icp.get_movement(self.pre_response, res, Ro, To )

        #R1 = Ro
        #t1 = To

        #print("正解")
        #print(acos(Ro[0,0]) / pi * 180)
        #print(To)
        print("ICP結果")
        if R1[0,0] > 1.0 or R1[0,0] < -1.0:
            R1[0,0] = 1.0 * R1[0,0] / abs(R1[0,0])
        print(acos(R1[0,0]) / pi * 180)
        print(t1)
        #print("誤差")
        #print(acos(R1[0,0]) / pi * 180-acos(Ro[0,0]) / pi * 180)
        #print(t1-To)

        self.R = R1.dot( self.R )
        #self.t = R1.dot( self.t ) + t1 
        #self.t =  self.t + t1 
        self.t = self.t + self.R.dot(t1)
        
        data_array = np.array(res)
        data_array = self.R.dot(data_array) + self.t

        self.img = self.plotPoint2Image( data_array )

        ##画像をarrayに変換
        #im_list = np.asarray(self.img)
        ##貼り付け
        #plt.imshow(im_list)
        #plt.gray()
        ##表示
        #plt.pause(.001)
        cv2.imshow("map", self.img)
        if cv2.waitKey(1) == 27:
            cv2.imwrite("map.jpg", self.img)


        #data_array = R1.dot(res)
        #data_array = np.array([data_array[0] + t1[0],
        #                       data_array[1] + t1[1]])

        #plt.scatter(data_array[0,:],data_array[1,:],marker = "o",color = "b",s = 10, label = "data3")

        #plt.legend(loc = "upper right")
        #plt.show()


        self.pre_response = res

        #self.img.show()
        #print (response)

    # map作成ループ
    def make_map_loop(self):
        while True:
            self.get_movement()
            self.get_lrf_data()


class LocalizationSimulator(CommMyRobotSimulator):
    def __init__(self, serverPort = 8000, serverAddr = 'localhost'):
        super().__init__(serverPort, serverAddr)

        self.pf = MyParticleFilter("./map2.jpg")

        self.init_pos = []
        self.init_dir = []
        self.now_pos = []
        self.now_dir = []
        
        self.pre_pos = []
        self.pre_dir = []

    # 現在地を取得し，メンバを更新
    def get_movement(self):
        response = super().get_movement()
        if self.init_pos == [] and self.init_dir == []:
            self.init_pos = response[0:3]
            self.init_dir = response[3:6]

            self.pf.init_positoin([300,
                                   1000],
                                  0.0)

            self.now_pos = response[0:3]
            self.now_dir = response[3:6]

        self.pre_pos = self.now_pos
        self.pre_dir = self.now_dir

        self.now_pos = response[0:3]
        self.now_dir = response[3:6]

        #rand_range = 0.5
        #rand_noise = random.random() * rand_range * 2 - rand_range
        #self.now_pos = [x + rand_noise for x in response[0:3]]
        #self.now_dir = [x + rand_noise for x in response[3:6]]

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        response = super().get_lrf_data()
        return self.calc_local_coord(response)

   #点群データから画像作成
    def plotPoint2Image(self, data):
        self.img = np.zeros((600, 600, 1), np.uint8)
        coefficient = 100.0 / 5.0
        origin_x = 300
        #origin_y = self.img.shape[1] / 2.0
        origin_y = 300
            
        for (raw_x, raw_y) in zip(data[0],data[1]):
            if raw_x == 0 and raw_y == 0:continue
            x = int(raw_x + origin_x)
            y = int(raw_y + origin_y)
        
            if self.img[y,x] != 250:
               self.img[y,x] = 200
        return self.img
     
    def localize(self):
        self.get_movement()
        self.pf.set_delta_position([self.now_pos[0]-self.pre_pos[0],self.now_pos[1]-self.pre_pos[1]],
                                   (self.now_dir[1]-self.pre_dir[1]) * pi / 180.0)
        lrf = self.pf.estimate_position(self.get_lrf_data())
        cv2.imshow("lrf",self.plotPoint2Image( lrf ))
        cv2.waitKey(0)


    def localization_loop(self):
        while True:
            self.localize()