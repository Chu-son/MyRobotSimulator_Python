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
import time


class CommMyRobotSimulator:
    def __init__(self, serverPort = 8000, serverAddr = "localhost"):
        self.serverAddr = serverAddr
        self.serverPort = serverPort

        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.connect((serverAddr, serverPort))

        self.urg_offset = [0.6,0.0]

        self._errorlist = []

    # 受信した点群データリストをロボット回転中心を原点とするxy座標に変換
    # 想定するデータリスト構造　:　([　radian ... ,　distance ... ])
    def calc_local_coord(self, dataList, remove_error = False):
        boundary = int( len(dataList) / 2)
        radian_list = dataList[:boundary]
        distance_list = dataList[boundary:len(dataList)]

        ret_x_list = []
        ret_y_list = []
        self._errorlist.clear()
        for ( rad , dist ) in zip( radian_list , distance_list ):
            if remove_error and (dist == 0.0 or dist == 0.1):continue

            if dist == 0.0:
                self._errorlist.append(True)
                dist = 10.0
            elif dist == 0.1:
                self._errorlist.append(True)
            else:
                self._errorlist.append(False)

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
        return self.get_data_list( "move get coordinate")

    def get_is_driving(self):
        return self.get_data_list( "move get isdriving")

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        return self.get_data_list("lrf a")

    # 駆動指令を送信
    def send_driving_command(self, direction, value, speed, tolerance):
        command = "move send direction " + direction + " " + str(value) + " " + str(speed) + " " + str(tolerance)
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

    def drive(self):
        d = DrivingSimulator()
        d.drive_follow_path()

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

    def line(self, img, start, end, color):
        x = start[0]
        y = start[1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        sx = 1 if end[0] > start[0] else -1
        sy = 1 if end[1] > start[1] else -1

        if dx >= dy:
            err = 2 * dy - dx
            for _ in range(dx+1):
                if img[y,x] == 0:
                    img[y,x] = color
                x += sx
                err += 2 * dy
                if err >= 0:
                    y += sy
                    err -= 2 * dx
        else:
            err = 2 * dx - dy
            for _ in range(dy+1):
                if img[y,x] == 0:
                    img[y,x] = color
                y += sy
                err += 2 * dx
                if err >= 0:
                    x += sx
                    err -= 2 * dy

        return img


    #点群データから画像作成
    def plotPoint2Image(self, data):
        coefficient = 100.0 / 5.0
        origin_x = self.img.shape[0] / 5.0
        #origin_y = self.img.shape[1] / 2.0
        origin_y = 1000
    
        index = -1
        for (raw_x, raw_y) in zip(data[0],data[1]):
            index += 1

            x = int(raw_x * coefficient + origin_x)
            y = int(raw_y * coefficient + origin_y)

            robot_x = self.now_pos[0]
            robot_y = self.now_pos[2]
            dir = -radians(self.now_dir[1] - self.init_dir[1])

            robot_x = int((robot_x + self.urg_offset[0] * cos( dir )) * coefficient + origin_x)
            robot_y = int(-(robot_y + self.urg_offset[0] * sin( dir )) * coefficient + origin_y)

            self.img = self.line(self.img, (robot_x,robot_y),(x,y),101)

            if self._errorlist[index] == True:continue
        
            if self.img[y,x] <= 200:
               self.img[y,x] += 50
            else :
                self.img[y,x] = 255
            #self.img[y,x] = 250 if (self.img[y,x] + 50) > 250 else (self.img[y,x] + 50)
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
        self.img = self.plotPoint2Image( self.calc_global_coord( self.calc_local_coord(response)))

        #plt.scatter(self.pre_response[0], self.pre_response[1], marker = "o",color = "r",s = 60, label = "data1")
        res = self.calc_local_coord(response)
        ##new_data, Ro, To = self.calc_odometry_correction_coord(res)
        ###plt.scatter(new_data[0,:],new_data[1,:],marker = "o",color = "g",s = 40, label = "data2")
        ##R1, t1, _ = self.icp.get_movement(self.pre_response, res, Ro, To )

        ###R1 = Ro
        ###t1 = To

        ###print("正解")
        ###print(acos(Ro[0,0]) / pi * 180)
        ###print(To)
        ##print("ICP結果")
        ##if R1[0,0] > 1.0 or R1[0,0] < -1.0:
        ##    R1[0,0] = 1.0 * R1[0,0] / abs(R1[0,0])
        ##print(acos(R1[0,0]) / pi * 180)
        ##print(t1)
        ###print("誤差")
        ###print(acos(R1[0,0]) / pi * 180-acos(Ro[0,0]) / pi * 180)
        ###print(t1-To)

        ##self.R = R1.dot( self.R )
        ###self.t = R1.dot( self.t ) + t1 
        ###self.t =  self.t + t1 
        ##self.t = self.t + self.R.dot(t1)
        
        ##data_array = np.array(res)
        ##data_array = self.R.dot(data_array) + self.t

        ##self.img = self.plotPoint2Image( data_array )

        ##画像をarrayに変換
        #im_list = np.asarray(self.img)
        ##貼り付け
        #plt.imshow(im_list)
        #plt.gray()
        ##表示
        #plt.pause(.001)
        cv2.imshow("map", self.img)
        if cv2.waitKey(1) == 27:
            cv2.imwrite("map.bmp", self.img)


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

        self.pf = MyParticleFilter("./map.bmp")
        self._show_img = cv2.imread("./map.bmp.jpg")

        # [m],[deg]
        self.init_pos = []
        self.init_dir = []

        self.now_pos = []
        self.now_dir = []
        self.now_pos_noisy = []
        self.now_dir_noisy = []
        
        self.pre_pos = []
        self.pre_dir = []
        self.pre_pos_noisy = []
        self.pre_dir_noisy = []

        self.est_pos = [] # [x,y,th]

        self.map_coord_origin = [300,1000]

    def _degree_disp(self, current_deg, pre_deg):
        disp = current_deg - pre_deg
        if disp >= 180.0:
            disp -= 360.0
        elif disp <= -180.0:
            disp += 360.0
        return disp


    # 現在地を取得し，メンバを更新
    def get_movement(self):
        response = super().get_movement()
        if self.init_pos == [] and self.init_dir == []:
            self.init_pos = response[0:3]
            self.init_dir = response[3:6]

            self.pf.init_positoin(self.map_coord_origin,
                                  0.0)

            self.now_pos = response[0:3]
            self.now_dir = response[3:6]

            self.now_pos_noisy = response[0:3]
            self.now_dir_noisy = response[3:6]

        self.pre_pos = self.now_pos
        self.pre_dir = self.now_dir

        self.pre_pos_noisy = self.now_pos_noisy
        self.pre_dir_noisy = self.now_dir_noisy

        self.now_pos = response[0:3]
        self.now_dir = response[3:6]

        rand_range = 1.0
        rand_noise = random.random() * rand_range * 2 - rand_range
        noisy_x = (self.now_pos[0] - self.pre_pos[0] + 0.1) * 1.1
        noisy_y = (self.now_pos[2] - self.pre_pos[2] + 0.1) * 1.1
        noisy_th = (self._degree_disp( self.now_dir[1], self.pre_dir[1]) - 2.5) * 1.1
        #self.now_pos_noisy = [x + rand_noise for x in response[0:3]]
        self.now_pos_noisy = [self.pre_pos_noisy[0] + noisy_x, self.pre_pos_noisy[1] + noisy_y, self.now_pos[2]]
        self.now_dir_noisy = [x + noisy_th for x in self.pre_dir_noisy]

    # 点群情報を取得し，map画像を作成
    def get_lrf_data(self):
        response = super().get_lrf_data()
        return self.calc_local_coord(response, True)

   #点群データから画像作成
    def plotPoint2Image(self, data):
        img = np.zeros((600, 600, 1), np.uint8)
        coefficient = 100.0 / 5.0
        origin_x = 300
        #origin_y = self.img.shape[1] / 2.0
        origin_y = 300
            
        for (raw_x, raw_y) in zip(data[0],data[1]):
            if raw_x == 0 and raw_y == 0:continue
            x = int(raw_x * coefficient + origin_x)
            y = int(raw_y * coefficient + origin_y)
        
            if img[y,x] != 250:
               img[y,x] = 200
        return img

    def _hsv_to_rgb(self, h, s, v):
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def show_position(self):
        #img = copy.deepcopy(self.pf._map_image)
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        img = copy.deepcopy(self._show_img)

        # パーティクルたち
        for i, p in enumerate( self.pf._particle_list ):
            c = self._hsv_to_rgb(int(i / self.pf._particle_num * 120),200,200) 
            #c = self._hsv_to_rgb(int(p.nomalized_weight*120),200,200) 
            #c = self._hsv_to_rgb(int(120),200,200) 
            #c = (0,255,0)
            cv2.circle(img,
                   (int(p.x),
                    int(p.y)),
                   1,c)

        for i, p in enumerate( self.pf._particle_list ):
            if p.nomalized_weight < 0.9:break
            c = self._hsv_to_rgb(int(i / self.pf._particle_num * 120),200,200) 
            #c = self._hsv_to_rgb(int(p.nomalized_weight*120),200,200) 
            #c = self._hsv_to_rgb(int(120),200,200) 
            #c = (0,255,0)
            cv2.circle(img,
                   (int(p.x),
                    int(p.y)),
                   7,(0,255,0),3)

        coeff = 1000 // 50
        # 真の位置
        cv2.circle(img,
                   (int((self.now_pos[0] - self.init_pos[0]) * coeff) + self.map_coord_origin[0],
                    int(-(self.now_pos[2] - self.init_pos[2]) * coeff) + self.map_coord_origin[1]),
                   10,(0,0,255),3)
        dir = radians(self.now_dir[1] - self.init_dir[1])
        cv2.line(img,
                 (int((self.now_pos[0] - self.init_pos[0]) * coeff) + self.map_coord_origin[0],
                    int(-(self.now_pos[2] - self.init_pos[2]) * coeff) + self.map_coord_origin[1]),
                 (int((self.now_pos[0] - self.init_pos[0]) * coeff + 15 * cos(dir)) + self.map_coord_origin[0],
                    int(-(self.now_pos[2] - self.init_pos[2]) * coeff + 15 * sin (dir)) + self.map_coord_origin[1]),
                    (0,0,255),3)
        # ノイズ入りの位置
        cv2.circle(img,
                   (int((self.now_pos_noisy[0] - self.init_pos[0]) * coeff) + self.map_coord_origin[0],
                    int(-(self.now_pos_noisy[1] - self.init_pos[1]) * coeff) + self.map_coord_origin[1]),
                   5,(0,0,255),3)
        dir = radians(self.now_dir_noisy[1] - self.init_dir[1])
        cv2.line(img,
                 (int((self.now_pos_noisy[0] - self.init_pos[0]) * coeff) + self.map_coord_origin[0],
                    int(-(self.now_pos_noisy[1] - self.init_pos[1]) * coeff) + self.map_coord_origin[1]),
                 (int((self.now_pos_noisy[0] - self.init_pos[0]) * coeff + 9 * cos(dir)) + self.map_coord_origin[0],
                    int(-(self.now_pos_noisy[1] - self.init_pos[1]) * coeff + 9 * sin (dir)) + self.map_coord_origin[1]),
                    (0,0,255),3)

        # 推定位置
        cv2.circle(img,
                   (int((self.est_pos[0]) * coeff) + self.map_coord_origin[0],
                    int((self.est_pos[1]) * coeff) + self.map_coord_origin[1]),
                   5,(255,0,0),3)
        cv2.line(img,
                 (int((self.est_pos[0]) * coeff) + self.map_coord_origin[0],
                    int((self.est_pos[1]) * coeff) + self.map_coord_origin[1]),
                 (int((self.est_pos[0]) * coeff + 9 * cos(self.est_pos[2])) + self.map_coord_origin[0],
                    int((self.est_pos[1]) * coeff + 9 * sin (self.est_pos[2])) + self.map_coord_origin[1]),
                    (255,0,0),3)

        cv2.imshow("estimate",img)

        th = self.pf._particle_list[0].theta
        R = np.array([[cos( -th ) , sin( -th)],
                      [ -sin( -th ), cos( -th)]])
        data = R.dot(self.lrf_data + np.array([[self.pf._particle_list[0].x - int((self.now_pos_noisy[0]) * coeff) - self.map_coord_origin[0]],
                                               [self.pf._particle_list[0].y - int((self.now_pos_noisy[1]) * coeff) - self.map_coord_origin[1]]])
                     )
        #data = R.dot(self.lrf_data )
        #cv2.imshow("particle",self.plotPoint2Image(data))
     
    def localize(self):
        self.get_movement()
        #self.pf.set_delta_position([self.now_pos[0]-self.pre_pos[0],self.now_pos[1]-self.pre_pos[1]],
        #                           (self.now_dir[1]-self.pre_dir[1]) * pi / 180.0)
        self.pf.set_delta_position([self.now_pos_noisy[0]-self.pre_pos_noisy[0],-(self.now_pos_noisy[1]-self.pre_pos_noisy[1])],
                                   self._degree_disp(self.now_dir_noisy[1],self.pre_dir_noisy[1]) * pi / 180.0)
        self.lrf_data = self.get_lrf_data()
        self.est_pos = self.pf.estimate_position(self.lrf_data)
        #cv2.imshow("lrf",self.plotPoint2Image( lrf ))
        self.show_position()
        cv2.waitKey(5)


    def localization_loop(self):
        while True:
            self.localize()


class DrivingSimulator(LocalizationSimulator):
    """
    1.経路情報読み込み
    2.回転
    3.直進
    stop送る式か座標送る式か
    """
    def __init__(self, serverPort = 8000, serverAddr = "localhost"):
        super().__init__(serverPort, serverAddr)

        self._route_file = open("map.bmp.rt",'r')
        self._route_file.readline() # ignore header

        self._pre_coord = [] # 画像座標
        self._now_coord = []
        self._next_coord = []

        self._get_next_coord()
        self._now_coord = [self._next_coord[0] - 5,
                           self._next_coord[1] ]


        self._orientation = 0.0 # [rad]

    def _get_next_coord(self):
        line = self._route_file.readline()
        if not line: return False

        line = [ int(x) for x in line.split(',')[:-1]]

        self._pre_coord = self._now_coord
        self._now_coord = self._next_coord

        self._next_coord = line[0:2]

        return True

    def _calc_rotate_angle(self):
        vec1 = [cos(self._orientation),sin(self._orientation)]
        vec2 = [ self._next_coord[0] - self._now_coord[0],
                self._next_coord[1] - self._now_coord[1]]

        absVec2 = sqrt(vec2[0]**2 + vec2[1]**2)
        vec2 = [vec2[0]/absVec2, vec2[1]/absVec2]

        det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        inner = vec1[0] * vec2[0] + vec1[1] * vec2[1]

        rad = atan2(det, inner)
        self._orientation += rad

        return rad

    def _calc_distance(self):
        disp = [ self._next_coord[0] - self._now_coord[0],
                 self._next_coord[1] - self._now_coord[1]]
        return sqrt( disp[0] ** 2 + disp[1] ** 2) * 50 / 1000 # [m]

    def _send_rotate_angle(self, radian):
        print("rotate" + str(degrees(radian)))
        if radian > 0:
            self.send_driving_command("right", degrees(radian), 5.0, 1.0)
        else:
            self.send_driving_command("left", degrees(radian), 5.0, 1.0) 

    def _send_straight_distance(self, distance):
        print("straight" + str(distance))
        if distance > 0:
            self.send_driving_command("forward", distance, 0.3, 1.0)
        else:
            self.send_driving_command("back", distance, 0.3, 1.0)

    def _drive(self):
        # rotate
        #self._send_rotate_angle(self._calc_rotate_angle() * self.pf._gaussian(1.0, 0.5))
        self._send_rotate_angle(self._calc_rotate_angle())
        while self.get_is_driving()[0]: 
            self.localize()
        
        # straight
        #self._send_straight_distance(self._calc_distance() * self.pf._gaussian(1.0, 0.5))
        self._send_straight_distance(self._calc_distance())
        while self.get_is_driving()[0]: 
            self.localize()

            #tmp_now = [self.now_pos,self.now_dir]
            #tmp_now_noisy = [self.now_pos_noisy,self.now_dir_noisy]

            #self.get_movement()
            #self._now_coord = [int(self.est_pos[0] * 1000 / 50) + self.map_coord_origin[0] + self.now_pos_noisy[0] - self.pre_pos_noisy[0],
            #                   int(-self.est_pos[1] * 1000 / 50) + self.map_coord_origin[1] + -(self.now_pos_noisy[2] - self.pre_pos_noisy[1])]
            #self._orientation = self.est_pos[2] + radians( self.now_dir_noisy[1] - self.pre_dir_noisy[1])

            #self.now_pos, self.now_dir = tmp_now
            #self.now_pos_noisy,self.now_dir_noisy = tmp_now_noisy

            #self._drive()

    def drive_follow_path(self):
        # next coord
        while self._get_next_coord():
            self._drive()