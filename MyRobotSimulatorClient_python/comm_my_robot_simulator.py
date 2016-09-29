import socket
import struct
from math import *
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from my_iterative_closest_point import *


class CommMyRobotSimulator:
    def __init__(self, serverPort = 8000, serverAddr = "localhost"):
        self.serverAddr = serverAddr
        self.serverPort = serverPort

        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.connect((serverAddr, serverPort))

        self.img = Image.new("L",(600,600))
        self.icp = MyIterativeClosestPoint()

        self.init_pos = []
        self.init_dir = []
        self.now_pos = []
        self.now_dir = []
        
        self.pre_pos = []
        self.pre_dir = []

        self.hit_count = 0
        self.pre_response = None

        self.R = np.identity(2)  #回転行列
        self.t = np.zeros([2,1])  #並進ベクトル

        self.urg_offset = [0.6,0.0]
        self.icp.error_data = self.urg_offset

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

    # 受信した現在の座標および姿勢から点群を世界座標に変換
    # 初めて受信した位置姿勢が基準となる
    def calc_global_coord(self, dataList):
        data_array = np.array(dataList)

        pos = [now - init for (now,init) in zip(self.now_pos,self.init_pos)]
        dir = [- radians(now - init) for (now,init) in zip(self.now_dir,self.init_dir)]

        R = numpy.array( [[ cos(dir[1]), sin(dir[1])],
                          [-sin(dir[1]), cos(dir[1])]] )
        T = numpy.array( [[ pos[0]] , [-pos[2]] ] )

        data_array = np.array([data_array[0],
                               data_array[1]])
        data_array = R.dot(data_array)
        data_array = np.array([data_array[0] + T[0],
                               data_array[1] + T[1]])
        return data_array

    def calc_odometry_correction_coord(self, dataList):
        data_array = np.array(dataList)

        pos = [now - pre for (pre, now, init) in zip(self.pre_pos, self.now_pos, self.init_pos)]
        dir = [- radians(now - pre) for (pre, now, init) in zip(self.pre_dir, self.now_dir, self.init_dir)]

        R = numpy.array( [[ cos(dir[1]), sin(dir[1])],
                          [-sin(dir[1]), cos(dir[1])]] )
        T = numpy.array( [[ pos[0]] , [-pos[2]] ] )

        data_array = np.array([data_array[0],
                               data_array[1]])
        data_array = R.dot(data_array)
        data_array = np.array([data_array[0] + T[0],
                               data_array[1] + T[1]])


        return data_array, R, T

    #点群データから画像作成
    def plotPoint2Image(self, data):
        imgMap = self.img.load()
        coefficient = 100.0 / 5.0
        origin_x = self.img.size[0] / 2.0
        origin_y = self.img.size[1] / 2.0
    
        for (raw_x, raw_y) in zip(data[0],data[1]):
            if raw_x == 0 and raw_y == 0:continue
            x = int(raw_x * coefficient + origin_x)
            y = int(raw_y * coefficient + origin_y)
        
            if imgMap[x,y] != 250:
               imgMap[x,y] += 50
        return self.img

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
        response = self.get_data_list( "move get")
        if self.init_pos == [] and self.init_dir == []:
            self.init_pos = response[0:3]
            self.init_dir = response[3:6]

            self.now_pos = response[0:3]
            self.now_dir = response[3:6]

        self.pre_pos = self.now_pos
        self.pre_dir = self.now_dir

        #self.now_pos = response[0:3]
        #self.now_dir = response[3:6]

        rand_range = 1.0
        rand_noise = random.random() * rand_range * 2 - rand_range
        self.now_pos = [x + rand_noise for x in response[0:3]]
        self.now_dir = [x + rand_noise for x in response[3:6]]

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        response = self.get_data_list("lrf a")
        if self.pre_response == None:
            self.pre_response = self.calc_local_coord(response)
            return
        #self.img = self.plotPoint2Image( self.calc_global_coord( self.calc_local_coord(response)))

        plt.scatter(self.pre_response[0], self.pre_response[1], marker = "o",color = "r",s = 60, label = "data1")
        res = self.calc_local_coord(response)
        new_data, Ro, To = self.calc_odometry_correction_coord(res)
        plt.scatter(new_data[0,:],new_data[1,:],marker = "o",color = "g",s = 40, label = "data2")
        R1, t1, _ = self.icp.get_movement(self.pre_response, res, Ro, To )

        #R1 = Ro
        #t1 = To

        #print("正解")
        #print(acos(Ro[0,0]) / pi * 180)
        #print(To)
        #print("ICP結果")
        #if R1[0,0] > 1.0 or R1[0,0] < -1.0:
        #    R1[0,0] = 1.0 * R1[0,0] / abs(R1[0,0])
        #print(acos(R1[0,0]) / pi * 180)
        #print(t1)
        #print("誤差")
        #print(acos(R1[0,0]) / pi * 180-acos(Ro[0,0]) / pi * 180)
        #print(t1-To)

        #self.R = R1.dot( self.R )
        #self.t = R1.dot( self.t ) + t1 
        
        #data_array = np.array(res)
        #data_array = np.array([data_array[0] ,
        #                       data_array[1]])
        #data_array = self.R.dot(data_array)
        #data_array = np.array([data_array[0] + self.t[0],
        #                       data_array[1] + self.t[1]])

        #self.img = self.plotPoint2Image( data_array )

        ##画像をarrayに変換
        #im_list = np.asarray(self.img)
        ##貼り付け
        #plt.imshow(im_list)
        #plt.gray()
        ##表示
        #plt.pause(.001)


        data_array = R1.dot(res)
        data_array = np.array([data_array[0] + t1[0],
                               data_array[1] + t1[1]])

        plt.scatter(data_array[0,:],data_array[1,:],marker = "o",color = "b",s = 10, label = "data3")

        plt.legend(loc = "upper right")
        plt.show()


        self.pre_response = res

        #self.img.show()
        #print (response)

    # 駆動指令を送信
    def send_driving_command(self, direction, speed):
        command = "move send " + direction + " " + str(speed)
        self.get_data_list(command)

    # 衝突しているオブジェクトの数を取得
    def get_hit_count(self):
        self.hit_count = self.get_data_list("hit")[0]

    # map作成ループ
    def make_map_loop(self):
        while True:
            self.get_movement()
            self.get_lrf_data()