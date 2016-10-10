import numpy as np
import cv2
from math import *
import random
import copy

class MyParticleFilter():
    class Particle():
        def __init__(self, x, y, theta, weight):
            self.x = x
            self.y = y
            self.theta = theta
            self.weight = weight
            self.nomalized_weight = 0.0

    def __init__(self, map_image_path):
        self._map_image = cv2.imread(map_image_path,0)
        self._ref_width = 25 * 1000 # 参照地図のサイズ(mm)
        self._ref_height = self._ref_width
        self._pixel_size = 50 # mm/pixel

        # 得点表作成
        self._points = [16,8,4,2,1]
        #self._points = [128,64,32,16,8,4,2,1]
        #self._points = [200,150,100,50,1] # 目視確認用
        self._point_len = len(self._points)-1
        self._point_table = []
        for y in range(len(self._points)*2-1):
            row = []
            for x in range(len(self._points)*2-1):
                row.append( self._points[max(abs(y-self._point_len),abs(x-self._point_len))] )
            self._point_table.append(row)

        # mm => pixel
        self._ref_width //= self._pixel_size
        self._ref_height //= self._pixel_size

        self._make_likelihood_map()

        self._particle_num = 500;
        self._particle_list = []

        self._pos = [0.0, 0.0] # [x, y] (pixel)
        self._theta = 0.0 # rad
        self._init_pos = [] # [x, y] (pixel)
        self._init_theta = 0.0 # (rad)

    # 事前地図から尤度マップ的なのを作成
    def _make_likelihood_map(self):
        self._likelihood_map = cv2.imread("./likelihood.jpg",0)
        return 

        print( "making likelihood map")
        ref_map = np.zeros_like(self._map_image)
        for y in range(self._point_len+1, self._map_image.shape[0]-self._point_len):
            for x in range(self._point_len+1, self._map_image.shape[1]-self._point_len):
                if self._map_image[y,x] > 100:
                    for yp in range(self._point_len * 2 + 1):
                        for xp in range(self._point_len * 2 + 1 ):
                            x_ref = x + (xp - self._point_len)
                            y_ref = y + (yp - self._point_len)
                            ref_map[y_ref,x_ref] = max(ref_map[y_ref,x_ref], self._point_table[yp][xp])

        #cv2.imshow("trim",trimed_map)
        #cv2.imshow("ref",ref_map)
        cv2.imwrite("likelihood.jpg",ref_map)
        self._likelihood_map = ref_map
        return ref_map

    def _prepare_ref_map(self):
        trimed_map = self._likelihood_map[max(self._pos[1] - self._ref_height/2,0) : self._pos[1] + self._ref_height/2,
                                  max(self._pos[0] - self._ref_width/2,0) : self._pos[0] + self._ref_width/2]        
        #cv2.imshow("trim",trimed_map)
        #cv2.imshow("ref",ref_map)
        #cv2.imwrite("ref.jpg",ref_map)
        self._ref_map = trimed_map
        return trimed_map

    # ガウス分布の乱数生成
    def _gaussian(self):
        x = 0.0
        while x == 0.0:
            x = random.random()
        y = random.random()
        s = sqrt( -2.0 * log(x) )
        t = 2.0 * pi * y

        return s * cos(t)

    # pos:[x,y](pixel), theta(rad)
    def init_positoin(self, pos, theta):
        self._init_pos = pos
        self._init_theta = theta

        self._pos = pos
        self._theta = theta        
        self._particle_list.clear()
        for _ in range(self._particle_num):
            self._particle_list.append( MyParticleFilter.Particle( pos[0] + self._gaussian() * 10,
                                                                   pos[1] + self._gaussian() * 10,
                                                                   theta + self._gaussian() * 0.005,
                                                                   0.0 ) )

    def _resample_particles(self):
        thre1 = 1/3
        thre2 = 1/3
        boundary1 = int(self._particle_num * thre1)
        boundary2 = int(self._particle_num * thre2) + boundary1
        var_fb = 20
        var_fb2 = 40
        var_ang = 0.005
        var_ang2 = 0.05

        #boundary1 = 0
        #for p in self._particle_list:
        #    if p.nomalized_weight < 0.6 or boundary1 > int(self._particle_num * thre1):break
        #    boundary1 += 1

        #boundary2 = int(self._particle_num * thre2) + boundary1

        for i in range(boundary1, boundary2):
            self._particle_list[i].x = self._particle_list[i - boundary1].x + self._gaussian() * var_fb
            self._particle_list[i].y = self._particle_list[i - boundary1].y + self._gaussian() * var_fb
            self._particle_list[i].theta = self._particle_list[i - boundary1].theta + self._gaussian() * var_ang

        for i in range(boundary2 , self._particle_num):
            #self._particle_list[i].x = self._particle_list[i - boundary].x + self._gaussian() * var_fb2
            #self._particle_list[i].y = self._particle_list[i - boundary].y + self._gaussian() * var_fb2
            #self._particle_list[i].theta = self._particle_list[i - boundary].theta + self._gaussian() * var_ang2
            self._particle_list[i].x = self._pos[0] + self._gaussian() * var_fb2
            self._particle_list[i].y = self._pos[1] + self._gaussian() * var_fb2
            self._particle_list[i].theta = self._theta + self._gaussian() * var_ang2
        
    # dPos : [x, y](m), dTh(rad)
    def set_delta_position(self, dPos, dTh):
        thre = 0.5
        num = int(self._particle_num * thre)
        #var_fb = 0.01
        var_fb = 10
        var_fb2 = 30
        var_ang = 0.005
        var_ang2 = 0.01

        # m => pixel
        dPos = [dPos[0] * 1000 / self._pixel_size, dPos[1] * 1000 / self._pixel_size]

        self._pos = [self._pos[0] + dPos[0], self._pos[1] + dPos[1]]
        self._theta += dTh

        #for i in range(num):
        #    #self._particle_list[i].x = self._particle_list[ i ].x + dPos[0] + self._gaussian() * var_fb
        #    #self._particle_list[i].y = self._particle_list[ i ].y + dPos[1] + self._gaussian() * var_fb
        #    #self._particle_list[i].theta = self._particle_list[ i ].theta + dTh + self._gaussian() * var_ang
        #    self._particle_list[i].x = self._pos[0] + self._gaussian() * var_fb
        #    self._particle_list[i].y = self._pos[1] + self._gaussian() * var_fb
        #    self._particle_list[i].theta = self._theta+ self._gaussian() * var_ang
        #for i in range(num, self._particle_num):
        #    #self._particle_list[i].x = self._particle_list[ i % num ].x + dPos[0] + cos( self._particle_list[ i % num ].theta) * self._gaussian() * var_fb
        #    #self._particle_list[i].y = self._particle_list[ i % num ].y + dPos[1] + sin( self._particle_list[ i % num ].theta) * self._gaussian() * var_fb
        #    #self._particle_list[i].x = self._particle_list[ i ].x + dPos[0] + self._gaussian() * var_fb
        #    #self._particle_list[i].y = self._particle_list[ i ].y + dPos[1] + self._gaussian() * var_fb
        #    #self._particle_list[i].theta = self._particle_list[ i ].theta + dTh + self._gaussian() * var_ang
        #    self._particle_list[i].x = self._pos[0] + self._gaussian() * var_fb2
        #    self._particle_list[i].y = self._pos[1] + self._gaussian() * var_fb2
        #    self._particle_list[i].theta = self._theta + self._gaussian() * var_ang2

        for i in range(self._particle_num):
            self._particle_list[i].x = self._particle_list[i].x + dPos[0]
            self._particle_list[i].y = self._particle_list[i].y + dPos[1]
            self._particle_list[i].theta = self._particle_list[i].theta + dTh

    def show_particles(self):
        img = copy.deepcopy(self._map_image)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        for p in self._particle_list:
            cv2.circle(img,
                   (int(p.x),
                    int(p.y)),
                   1,(0,255,0))

        cv2.circle(img,
                   (int(self._pos[0]),
                    int(self._pos[1])),
                   5,(0,0,255))

        cv2.imshow("estimate",img)

        cv2.waitKey(0)

    #点群データから画像作成
    def plotPoint2Image(self, data):
        img = np.zeros((600, 600, 1), np.uint8)
        origin_x = 300
        origin_y = 300
            
        for (raw_x, raw_y) in zip(data[0],data[1]):
            if raw_x == 0 and raw_y == 0:continue
            x = int(raw_x + origin_x)
            y = int(raw_y + origin_y)
        
            if img[y,x] != 250:
               img[y,x] = 200
        cv2.imshow("particle_view",img)

    def _calc_particle_weight(self, particle, lrfdata, refmap):
        R = np.array([[cos( -particle.theta ) , sin( -particle.theta)],
                      [ -sin( -particle.theta ), cos( -particle.theta)]])
        T = np.array([[particle.x - self._pos[0]],
                      [particle.y - self._pos[1]]])
        data = R.dot(lrfdata + T)

        #self.plotPoint2Image(data)
        #cv2.waitKey(20)

        ret_weight = 0
        for x, y in zip( data[0], data[1] ):
            y = int(y + self._ref_height/2 + 0.5)
            if y < 0: y = 0
            elif y >= refmap.shape[1]: y = refmap.shape[1] - 1
            x = int(x + self._ref_width/2 + 0.5)
            if x < 0: x = 0
            elif x >= refmap.shape[0]: x = refmap.shape[0] - 1
            ret_weight += refmap[y,x]

        return ret_weight

    def __calc_pos_ave(self):
        num = 0
        x = 0.0
        y = 0.0
        th = 0.0
        for p in self._particle_list:
            if p.weight != self._particle_list[0].weight: break
            num += 1
            x += p.x
            y += p.y
            th += p.theta
        x = x/num
        y = y/num
        th = th/num

        return [x, y], th

    def __calc_pos_weight_ave(self,lrfdata):
        num = 0
        w = 0.0
        x = 0.0
        y = 0.0
        th = 0.0
        for p in self._particle_list:
            if p.nomalized_weight < 0.8:break
            num += 1
            w += p.weight
            x += p.x * p.weight
            y += p.y * p.weight
            th += p.theta * p.weight
        x = x/w
        y = y/w
        th = th/w

        acc = w / (num * self._points[0] * len(lrfdata[0]))

        return [x, y], th, acc

    def __normalize_weight(self):
        max_val = self._particle_list[0].weight
        for i in range(self._particle_num):
            self._particle_list[i].nomalized_weight = self._particle_list[i].weight / max_val

    # LRFdata : [ [ x0, x1, x2 ...], [y0, y1, y2, ...] ](m)
    def estimate_position(self, LRFdata):
        #print("start estimate")
        ref_map = self._prepare_ref_map()

        # m => pixel
        lrfdata = np.array(LRFdata) * 1000 / self._pixel_size

        for index in range(self._particle_num):
            self._particle_list[index].weight = self._calc_particle_weight(self._particle_list[index],
                                                                           lrfdata,
                                                                           ref_map)
        # ソート
        self._particle_list = sorted(self._particle_list, key = lambda x : x.weight, reverse = True)
        self.__normalize_weight()

        #self._pos = [self._particle_list[0].x, self._particle_list[0].y]
        #self._theta = self._particle_list[0].theta
        #self._pos, self._theta = self.__calc_pos_ave()
        p,th, acc = self.__calc_pos_weight_ave(lrfdata)
        print(acc * 100.0)

        if acc > 0.5:
            self._pos, self._theta = p,th
        #self.show_particles()

        #print("end estimate")
        self._resample_particles()
        return ((self._pos[0] - self._init_pos[0]) * self._pixel_size / 1000 ,
                (self._pos[1] - self._init_pos[1]) * self._pixel_size / 1000 ,
                self._theta)
