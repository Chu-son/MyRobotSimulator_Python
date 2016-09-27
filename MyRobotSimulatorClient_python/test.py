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
    nPoint = 500  #���[�U�_�̐�
    fieldLength = 500   #�_���΂�T���ő勗��
    motion = np.array([10, 15, 30])   #�^�̈ړ���[���ix[m],���iy[m],��][deg]]
    transitionSigma = 1    #���i�����̈ړ��덷�W���΍�[m]
    thetaSigma = 1    #��]�����̌덷�W���΍�[deg]

    # �_�������_���ł΂�T��(t - 1�̎��̓_�Q)
    # data1:2�snPoint��
    data1 = fieldLength * rand(2,nPoint) - fieldLength / 2
    plt.scatter(data1[0,:],data1[1,:],marker = "o",color = "r",s = 60, label = "data1(before matching)")

    # data2 =  data1���ړ������� & �m�C�Y�t��
    # ��]���� �� �m�C�Y�t��
    theta = icp.toRadian(motion[2]) + icp.toRadian(thetaSigma) * rand()

    # ���i�x�N�g�� �� �m�C�Y�t��
    t = np.matlib.repmat(icp.transposition( motion[0:2] ),1,nPoint) + transitionSigma * randn(2,nPoint)

    # ��]�s��̍쐬
    A = np.array([[cos(theta), sin(theta)],
                 [-sin(theta), cos(theta)]])

    # data1���ړ�������data2�����
    data2 = t + A.dot(data1)
    plt.scatter(data2[0,:],data2[1,:],marker = "x",color = "b",s = 60, label = "data2")

    # ICP�A���S���Y�� data2��data1��Matching
    # R:��]�s��@t:���i�x�N�g��
    # R,T = icp(data1,data2)
    R = np.identity(2)  #��]�s��
    t = np.zeros([2,1])  #���i�x�N�g��
    R,T,matchData = icp.get_movement(data2, data1, R, t)
    plt.scatter(matchData[0,:],matchData[1,:],marker = "o",color = "g",s = 60, label = "data1(after matching)")

    #���ʂ̕\��
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