import numpy as np
import cv2
from math import *
import random

class MyParticleFilter():
    class Particle():
        def __init__(self, x, y, theta, weight):
            self.x = x
            self.y = y
            self.theta = theta
            self.weight = weight

    def __init__(self, map_image_path):
        self._map_image = cv2.imread(map_image_path,0)
        self._ref_width = 30 * 1000 # 参照地図のサイズ(mm)
        self._ref_height = self._ref_width
        self._pixel_size = 50 # mm/pixel

        # 得点表作成
        self._points = [16,8,4,2,1]
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

        self._particle_num = 500;
        self._particle_list = []

        self._pos = [0.0, 0.0] # [x, y] (pixel)
        self._theta = 0.0 # rad
        self._init_pos = [] # [x, y] (pixel)
        self._init_theta = 0.0 # (rad)

    # 事前地図から尤度マップ的なのを作成
    def _prepare_ref_map(self):
        trimed_map = self._map_image[self._pos[1] - self._ref_height/2 : self._pos[1] + self._ref_height/2,
                                  self._pos[0] - self._ref_width/2 : self._pos[0] + self._ref_width/2]        
        ref_map = np.zeros_like(trimed_map)
        for y in range(self._point_len+1, self._ref_height-self._point_len):
            for x in range(self._point_len+1, self._ref_width-self._point_len):
                if trimed_map[y,x] > 100:
                    for yp in range(self._point_len * 2 + 1):
                        for xp in range(self._point_len * 2 + 1 ):
                            x_ref = x + (xp - self._point_len)
                            y_ref = y + (yp - self._point_len)
                            ref_map[y_ref,x_ref] = max(ref_map[y_ref,x_ref], self._point_table[yp][xp])

        cv2.imshow("trim",trimed_map)
        cv2.imshow("ref",ref_map)
        #cv2.imwrite("ref.jpg",ref_map)
        return ref_map

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
            self._particle_list.append( MyParticleFilter.Particle( pos[0], pos[1], theta, 0.0 ) )
        
    # dPos : [x, y](m), dTh(rad)
    def set_delta_position(self, dPos, dTh):
        thre = 0.5
        num = int(self._particle_num * thre)
        var_fb = 0.01
        var_ang = 0.005

        dPos = [dPos[0] * 1000 / self._pixel_size, dPos[1] * 1000 / self._pixel_size]

        self._pos = [self._pos[0] + dPos[0], self._pos[1] + dPos[1]]
        self._theta += dTh

        for i in range(self._particle_num):
            self._particle_list[i].x = self._particle_list[ i % num ].x + dPos[0] + cos( self._particle_list[ i % num ].theta) * self._gaussian() * var_fb
            self._particle_list[i].y = self._particle_list[ i % num ].y + dPos[1] + sin( self._particle_list[ i % num ].theta) * self._gaussian() * var_fb
            self._particle_list[i].theta = self._particle_list[ i % num ].theta + dTh + self._gaussian() * var_ang

    def set_measured_data(self, data):
        pass

    def _calc_particle_waight(self, particle, lrfdata, refmap):
        R = np.array([cos( -particle.theta ) , sin( -particle.theta)],
                     [ -sin( -particle.theta ), cos( -particle.theta)])
        T = np.array([[particle.x],
                      [particle.y]])
        data = R.dot(lrfdata + T)

        ret_waight = 0
        for x, y in zip( data ):
            ret_waight += refmap[y,x]

        return ret_waight

    # LRFdata : [ [ x0, x1, x2 ...], [y0, y1, y2, ...] ](m)
    def estimate_position(self, LRFdata):
        ref_map = self._prepare_ref_map()

        # m => pixel
        LRFdata = np.array(LRFdata) * 1000 / self._pixel_size
        T = np.array([[self._pos[0]],[self._pos[1]]]) 
        #LRFdata = LRFdata + T
        R  = np.array([[cos(-self._theta), sin(-self._theta)],
                       [-sin(-self._theta), cos(-self._theta)]])
        lrfdata = R.dot(LRFdata)

        for index in range(self._particle_num):
            self._particle_list[index].weight = self._calc_particle_waight(self._particle_list[index],
                                                                           lrfdata,
                                                                           ref_map)
        # ソート
        
        return lrfdata
