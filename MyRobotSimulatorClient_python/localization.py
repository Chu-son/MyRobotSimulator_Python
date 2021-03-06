﻿import numpy as np
import cv2
from math import *
import random
import copy
from multiprocessing import pool
from multiprocessing import process
from processing_timer import ProcessingTimer

class MyParticleFilter():
    class Particle():
        def __init__(self, x, y, theta, weight):
            self.x = x # マップ座標[pixel]
            self.y = y
            self.theta = theta # [rad]
            self.weight = weight # 重み
            self.nomalized_weight = 0.0 # 0~1.0に正規化した重み

    def __init__(self, map_image_path):
        self._map_image = cv2.imread(map_image_path,0) # マップ画像
        self._ref_width = 25 * 1000 # 参照地図のサイズ(mm)
        self._ref_height = self._ref_width
        self._pixel_size = 50 # mm/pixel

        # 得点表作成
        self._points = [64,32,16,4,2]
        self._no_obstacle_point = 2
        #self._points = [255,128,32,8]
        #self._points = [200,150,100,50,1] # 目視確認用
        #self._no_obstacle_point = 16

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

        # 尤度マップ作製
        self._make_likelihood_map()

        # パーティクル
        self._particle_num = 2000;
        self._particle_list = []

        self._pos = [0.0, 0.0] # [x, y] (pixel)
        self._theta = 0.0 # rad
        self._init_pos = [] # [x, y] (pixel)
        self._init_theta = 0.0 # (rad)

        self._is_init_gpu = False

    # 事前地図から尤度マップ的なのを作成
    def _make_likelihood_map(self):
        self._likelihood_map = cv2.imread("./likelihood.bmp",0)
        if self._likelihood_map is not None: return 

        print( "making likelihood map")
        ref_map = np.zeros_like(self._map_image)
        for y in range(self._point_len+1, self._map_image.shape[0]-self._point_len):
            for x in range(self._point_len+1, self._map_image.shape[1]-self._point_len):
                if self._map_image[y,x] == 101:
                    ref_map[y,x] = self._no_obstacle_point

                if self._map_image[y,x] > 101:
                    for yp in range(self._point_len * 2 + 1):
                        for xp in range(self._point_len * 2 + 1 ):
                            x_ref = x + (xp - self._point_len)
                            y_ref = y + (yp - self._point_len)
                            ref_map[y_ref,x_ref] = max(ref_map[y_ref,x_ref], self._point_table[yp][xp])

        #cv2.imshow("trim",trimed_map)
        #cv2.imshow("ref",ref_map)
        cv2.imwrite("likelihood.bmp",ref_map)
        self._likelihood_map = ref_map
        return ref_map

    # 尤度マップから現在地に基づく参照マップを切り取り
    def _prepare_ref_map(self):

        # とりあえず
        self._pos = list(map(int,self._pos))

        trimed_map = self._likelihood_map[max(self._pos[1] - self._ref_height//2,0) : self._pos[1] + self._ref_height//2,
                                  max(self._pos[0] - self._ref_width//2,0) : self._pos[0] + self._ref_width//2]        
        #cv2.imshow("trim",trimed_map)
        #cv2.imshow("ref",ref_map)
        #cv2.imwrite("ref.jpg",ref_map)
        self._ref_map = trimed_map
        return trimed_map

    # ガウス分布の乱数生成
    def _gaussian(self, mu, sigma):
        x = 0.0
        while x == 0.0:
            x = np.random.random()
        y = np.random.random()
        s = sqrt( -2.0 * log(x) )
        t = 2.0 * pi * y

        return mu + sigma * s * sin(t)

    # 初期位置設定
    # pos:[x,y](pixel), theta(rad)
    def init_positoin(self, pos, theta):
        self._init_pos = pos
        self._init_theta = theta

        self._pos = pos
        self._theta = theta        
        self._particle_list.clear()
        for _ in range(self._particle_num):
            self._particle_list.append( MyParticleFilter.Particle( pos[0] + self._gaussian(0, 30),
                                                                   pos[1] + self._gaussian(0, 30),
                                                                   theta + self._gaussian(0, 0.1),
                                                                   0.0 ) )

    # 指定した分散に基づいてパーティクルをランダム移動
    def _get_resample_particle(self, particle, sigma_pos, sigma_theta):
        #p = copy.deepcopy(particle)
        p = MyParticleFilter.Particle(particle.x, particle.y, particle.theta,0.0)
        p.x = int(particle.x + self._gaussian(0, sigma_pos))
        p.y = int(particle.y + self._gaussian(0, sigma_pos))
        p.theta = particle.theta + self._gaussian(0, sigma_theta)

        # 尤度マップ的にあり得ない位置なら再計算
        return p if self._likelihood_map[p.y,p.x] != 0 \
                    else self._get_resample_particle(particle,sigma_pos,sigma_theta)

    # パーティクル群を分散σに基づいてランダムサンプリング
    def _random_sample_particles(self, sigma_pos, sigma_theta):
        for i in range(self._particle_num):
            self._particle_list[i] = self._get_resample_particle( MyParticleFilter.Particle(self._pos[0],
                                                                                            self._pos[1],
                                                                                            self._theta,
                                                                                            0.0),
                                                                 sigma_pos,
                                                                 sigma_theta)

    # パーティクル群のリサンプリング
    def _resample_particles(self):
        thre1 = 1/50
        thre2 = 1 - thre1 - 1/3
        boundary1 = int(self._particle_num * thre1)
        boundary2 = int(self._particle_num * thre2) + boundary1

        var_fb = 0.5 * 1000 // self._pixel_size
        var_fb2 = 2.5 * 1000 // self._pixel_size
        var_ang = 3 * pi / 180 # [rad]
        var_ang2 = 60 * pi / 180

        #boundary1 = 0
        #for p in self._particle_list:
        #    if p.nomalized_weight < 0.8 or boundary1 > int(self._particle_num * thre1):break
        #    boundary1 += 1

        #boundary2 = int(self._particle_num * thre2) + boundary1

        for i in range(boundary1, boundary2):
            self._particle_list[i] = self._get_resample_particle( self._particle_list[i % boundary1],
                                                                 var_fb, 
                                                                 var_ang)

        for i in range(boundary2 , self._particle_num):
            #self._particle_list[i].x = self._particle_list[i - boundary].x + self._gaussian() * var_fb2
            #self._particle_list[i].y = self._particle_list[i - boundary].y + self._gaussian() * var_fb2
            #self._particle_list[i].theta = self._particle_list[i - boundary].theta + self._gaussian() * var_ang2
            self._particle_list[i] = self._get_resample_particle( MyParticleFilter.Particle( self._pos[0], self._pos[1], self._theta, 0.0),
                                                                 var_fb2,
                                                                 var_ang2)
        
    # オドメトリによる推定移動量の設定
    # dPos : [x, y](m), dTh(rad)
    def set_delta_position(self, dPos, dTh):

        # m => pixel
        dPos = [dPos[0] * 1000 / self._pixel_size, dPos[1] * 1000 / self._pixel_size]

        self._pos = [self._pos[0] + dPos[0], self._pos[1] + dPos[1]]
        self._theta += dTh

        sigma_pos = 0.5
        sigma_th = 0.5
        for i in range(self._particle_num):
            self._particle_list[i].x = self._particle_list[i].x + dPos[0] * (self._gaussian(1.0, sigma_pos))
            self._particle_list[i].y = self._particle_list[i].y + dPos[1] * (self._gaussian(1.0, sigma_pos))
            self._particle_list[i].theta = self._particle_list[i].theta + dTh * (self._gaussian(1.0, sigma_th))

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

    # パーティクルの重みを計算
    def _calc_particle_weight(self, particle, lrfdata, refmap):
        #return 0
        if self._likelihood_map[int(particle.y),int(particle.x)] == 0:
            return 0

        R = np.array([[cos( -particle.theta ) , sin( -particle.theta)],
                      [ -sin( -particle.theta ), cos( -particle.theta)]])
        T = np.array([[particle.x - self._pos[0]],
                      [particle.y - self._pos[1]]])
        data = R.dot(lrfdata) + T

        #self.plotPoint2Image(data)
        #cv2.waitKey(20)

        ret_weight = 0
        for x, y in zip( data[0], data[1] ):
            #y = int(y + self._ref_height/2 + 0.5)
            #if y < 0: y = 0
            #elif y >= refmap.shape[1]: y = refmap.shape[1] - 1
            #x = int(x + self._ref_width/2 + 0.5)
            #if x < 0: x = 0
            #elif x >= refmap.shape[0]: x = refmap.shape[0] - 1

            y = int(y + self._ref_height/2)
            if y < 0 or y >= refmap.shape[1]: continue
            x = int(x + self._ref_width/2)
            if x < 0 or x >= refmap.shape[0]: continue

            ret_weight += refmap.item(y,x)

            #y = int(y + self._ref_height/2)
            #if y < 0 or y >= len(refmap): continue
            #x = int(x + self._ref_width/2)
            #if x < 0 or x >= len(refmap[0]): continue

            #ret_weight += refmap[y][x]

        return ret_weight

    def _calc_particle_weight_multi(self, args):
        return self._calc_particle_weight(args[0],args[1],args[2])

    def _calc_particle_weight_multi2(self, args):
        return [ self._calc_particle_weight(arg[0],arg[1],arg[2]) for arg in args]

    # 重み最大のパーティクルの平均で位置を推定
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

    # 重み付き平均で位置を推定
    def __calc_pos_weight_ave(self,lrfdata):
        num = 0
        w = 0.0
        x = 0.0
        y = 0.0
        th = 0.0
        for p in self._particle_list:
            if p.nomalized_weight < 0.9:break
            num += 1
            w += p.weight
            x += p.x * p.weight
            y += p.y * p.weight
            th += p.theta * p.weight
        if w == 0:
            x = self._particle_list[0].x
            y = self._particle_list[0].y
            th = self._particle_list[0].theta

            acc = 0.0

        else:
            x = x/w
            y = y/w
            th = th/w

            acc = w / (num * self._points[0] * len(lrfdata[0]))
        
        return [x, y], th, acc

    # 重みを正規化
    def __normalize_weight(self):
        max_val = self._particle_list[0].weight if self._particle_list[0].weight != 0 else 1
        for i in range(self._particle_num):
            self._particle_list[i].nomalized_weight = self._particle_list[i].weight / max_val

    def __init_gpu(self):
        if self._is_init_gpu:return

        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        self.__calc_weight_func = SourceModule(
            """
            __global__ void calcWeight(float** pxy, float** lrfdata, float** refmap)
            {
            }
            """)

        self._is_init_gpu = True

    def _calc_particle_weight_gpu(self, particles, lrfdata, refmap):
        particles_x_array = np.array([p.x for p in particles], dtype = np.float32)
        particles_y_array = np.array([p.y for p in particles], dtype = np.float32)


    # 自己位置推定
    # LRFdata : [ [ x0, x1, x2 ...], [y0, y1, y2, ...] ](m)
    def estimate_position(self, LRFdata):
        # 処理時間計測
        with ProcessingTimer("localize time"):

            #print("start estimate")

            # 参照マップ作製
            ref_map = self._prepare_ref_map()
            #cv2.imshow("ref",self._ref_map)

            # m => pixel
            lrfdata = np.array(LRFdata) * 1000 / self._pixel_size
            #self.plotPoint2Image(lrfdata)

            # 重み計算
            #for index in range(self._particle_num):
            #    self._particle_list[index].weight = self._calc_particle_weight(self._particle_list[index],
            #                                                                   lrfdata,
            #                                                                   ref_map)

            args = [[p, lrfdata, ref_map] for p in self._particle_list]
            pool_num = 6

            args_split_num = int(len(args)/pool_num + 0.5)
            args = [args[x:x+args_split_num] for x in range(0,len(args),args_split_num)]

            p = pool.Pool(pool_num)
            with ProcessingTimer("calc weight time"):
                result = list(p.map(self._calc_particle_weight_multi2,args))
            index = 0
            for weight_list in result:
                for w in weight_list:
                    self._particle_list[index].weight = w
                    index += 1
            p.close()

            #p = pool.Pool(pool_num)
            #with ProcessingTimer("calc weight time"):
            #    result = list(p.map(self._calc_particle_weight_multi,args))
            #for index, w in enumerate(result):
            #    self._particle_list[index].weight = w
            #p.close()

            # ソート
            self._particle_list = sorted(self._particle_list, key = lambda x : x.weight, reverse = True)
            # 重み正規化
            self.__normalize_weight()

            # 重み最大のパーティクルをロボットの位置と推定する場合
            #self._pos = [self._particle_list[0].x, self._particle_list[0].y]
            #self._theta = self._particle_list[0].theta

            # 重み最大のパーティクル群の平均を推定位置とする場合
            #self._pos, self._theta = self.__calc_pos_ave()
            #self._resample_particles()

            # 重み平均に基づいて位置推定
            p,th, acc = self.__calc_pos_weight_ave(lrfdata)
            w = self._calc_particle_weight(MyParticleFilter.Particle(p[0],p[1],th,0),
                                           lrfdata,ref_map)
            acc = w / (self._points[0] * len(lrfdata[0])) if len(lrfdata[0]) != 0 else 0
            print(acc * 100.0)
            if acc >= 0:
                self._pos, self._theta = p,th
                self._resample_particles()
            else:
                self._random_sample_particles(50, 0.1)

            #self.show_particles()

            #print("end estimate")

        return ((self._pos[0] - self._init_pos[0]) * self._pixel_size / 1000 ,
                (self._pos[1] - self._init_pos[1]) * self._pixel_size / 1000 ,
                self._theta)
