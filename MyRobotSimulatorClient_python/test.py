# -*- coding:utf-8 -*-

from math import *
import numpy as np
import numpy.matlib
from numpy.random import *
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from my_iterative_closest_point import *

def ICPsample():
    icp = MyIterativeClosestPoint()

    #Simulation Parameters
    nPoint = 500  #レーザ点の数
    fieldLength = 500   #点をばら撒く最大距離
    motion = np.array([10, 15, 30])   #真の移動量[並進x[m],並進y[m],回転[deg]]
    transitionSigma = 1    #並進方向の移動誤差標準偏差[m]
    thetaSigma = 1    #回転方向の誤差標準偏差[deg]

    # 点をランダムでばら撒く(t - 1の時の点群)
    # data1:2行nPoint列
    data1 = fieldLength * rand(2,nPoint) - fieldLength / 2
    plt.scatter(data1[0,:],data1[1,:],marker = "o",color = "r",s = 60, label = "data1(before matching)")

    # data2 =  data1を移動させる & ノイズ付加
    # 回転方向 ＆ ノイズ付加
    theta = icp.toRadian(motion[2]) + icp.toRadian(thetaSigma) * rand()

    # 並進ベクトル ＆ ノイズ付加
    t = np.matlib.repmat(icp.transposition( motion[0:2] ),1,nPoint) + transitionSigma * randn(2,nPoint)

    # 回転行列の作成
    A = np.array([[cos(theta), sin(theta)],
                 [-sin(theta), cos(theta)]])

    # data1を移動させてdata2を作る
    data2 = t + A.dot(data1)
    plt.scatter(data2[0,:],data2[1,:],marker = "x",color = "b",s = 60, label = "data2")

    # ICPアルゴリズム data2とdata1のMatching
    # R:回転行列　t:併進ベクトル
    # R,T = icp(data1,data2)
    R = np.identity(2)  #回転行列
    t = np.zeros([2,1])  #並進ベクトル
    R,T,matchData = icp.get_movement(data2, data1, R, t)
    plt.scatter(matchData[0,:],matchData[1,:],marker = "o",color = "g",s = 60, label = "data1(after matching)")

    #結果の表示
    print('True Motion [m m deg]:')
    print( motion )

    print('Estimated Motion [m m deg]:')
    theta  =  acos(R[0,0]) / pi * 180
    Est = np.hstack([icp.transposition(T), icp.transposition( np.array([theta]))])
    print("{:.4f}, {:.4f}, {:.4f}".format(Est[0][0],Est[0][1],Est[0][2]))

    print('Error [m m deg]:')
    Error = Est - motion
    print( "{:.4f}, {:.4f}, {:.4f}".format(Error[0][0],Error[0][1],Error[0][2]) )

    plt.grid(True)
    plt.legend(loc = "upper right")
    plt.show()

ICPsample()