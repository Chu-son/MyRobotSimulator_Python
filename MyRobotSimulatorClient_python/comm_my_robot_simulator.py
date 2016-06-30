import socket
import struct
from math import *
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

        self.img = Image.new("L",(400,400))
        self.icp = MyIterativeClosestPoint()

        self.init_pos = []
        self.init_dir = []
        self.now_pos = []
        self.now_dir = []

        self.hit_count = 0

    # 受信した点群データリストをLRF中心のxy座標に変換
    # 想定するデータリスト構造　:　([　radian ... ,　distance ... ])
    def calc_local_coord(self, dataList):
        boundary = int( len(dataList) / 2)
        radian_list = dataList[:boundary]
        distance_list = dataList[boundary:len(dataList)]

        ret_x_list = []
        ret_y_list = []

        for ( rad , dist ) in zip( radian_list , distance_list ):
            ret_x_list.append( dist * cos( rad ) )
            ret_y_list.append( - dist * sin( rad ) )

        return ret_x_list , ret_y_list

    # 受信した現在の座標および姿勢から点群を世界座標に変換
    # 初めて受信した位置姿勢が基準となる
    def calc_global_coord(self, dataList):
        data_array = np.array(dataList)

        pos = [now - init for (now,init) in zip(self.now_pos,self.init_pos)]
        dir = [-radians(now - init) for (now,init) in zip(self.now_dir,self.init_dir)]

        R = numpy.array( [[ cos(dir[1]), sin(dir[1])],
                          [-sin(dir[1]), cos(dir[1])]] )
        T = numpy.array( [[ pos[0]] , [-pos[2]] ] )

        data_array = np.array([data_array[0] + 0.6,
                               data_array[1]])
        data_array = R.dot(data_array)
        data_array = np.array([data_array[0] + T[0],
                               data_array[1] + T[1]])

        return data_array


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

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        response = self.get_data_list("lrf a")
        #img = plotPoint2Image( icp.ICPMatching( calc_local_coord(response)) , img)
        self.img = self.plotPoint2Image( self.calc_global_coord( self.calc_local_coord(response)))

        #画像をarrayに変換
        im_list = np.asarray(self.img)
        #貼り付け
        plt.imshow(im_list)
        plt.gray()
        #表示
        plt.pause(.001)

        #img.show()
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