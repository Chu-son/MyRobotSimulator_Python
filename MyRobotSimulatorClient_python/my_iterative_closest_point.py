from math import *
import numpy as np
import numpy.matlib

class MyIterativeClosestPoint:

    def __init__(self):
        self.__pre_data_array = np.array([])
        self.__new_data_array = np.array([])

        self.__pre_boundaryList = []
        self.__now_boundaryList = []

        self.__pre_gradientList = []
        self.__now_gradientList = []

        self.preR = np.identity(2)
        self.preT = np.zeros([2,1])

        self.is_init = False


    # ICPアルゴリズムによる、並進ベクトルと回転行列の計算を実施する関数
    # data1  =  [x(t)1 x(t)2 x(t)3 ...]
    # data2  =  [x(t + 1)1 x(t + 1)2 x(t + 1)3 ...]
    # x = [x y z]'
    def ICPMatching(self, next_data):

        next_data = np.array(next_data)
        self.__pre_data_array = self.__new_data_array
        self.__new_data_array = next_data

        self.make_boundary_list()
        self.make_gradient_list()

        if not self.is_init:
            self.is_init = True
            return self.__new_data_array

        #ICP パラメータ
        preError = 0    #一つ前のイタレーションのerror値
        dError = 10000   #エラー値の差分
        EPS = 0.0001    #収束判定値
        maxIter = 500   #最大イタレーション数
        count = 0       #ループカウンタ

        R = np.identity(2)  #回転行列
        t = np.zeros([2,1])  #並進ベクトル

        #初期位置合わせ
        self.__new_data_array = self.preR.dot( self.__new_data_array )
        self.__new_data_array = np.array([self.__new_data_array[0] + self.preT[0],
                                          self.__new_data_array[1] + self.preT[1]])

        while not(dError < EPS):
        #while False:
            count = count + 1
    
            ii, error = self.FindNearestPoint( )  #最近傍点探索
            R1, t1 = self.SVDMotionEstimation( ii )    #特異値分解による移動量推定

            #計算したRとtで点群とRとtの値を更新
            self.__new_data_array = R1.dot( self.__new_data_array )
            self.__new_data_array = np.array([self.__new_data_array[0] + t1[0],
                                              self.__new_data_array[1] + t1[1]])
            R = R1.dot( R )
            t = R1.dot( t ) + t1 
    
            dError = abs(preError - error)  #エラーの改善量
            preError = error    #一つ前のエラーの総和値を保存
    
            if count > maxIter:  #収束しなかった
                print('Max Iteration')
                return

        print('Convergence:' + str( count ))

        self.preR = R.dot( self.preR )
        self.preT = R.dot( self.preT ) + t 
    
        return self.__new_data_array

    #data2に対するdata1の最近傍点のインデックスを計算する関数
    def FindNearestPoint(self):
        m1 = self.__pre_data_array.shape[1]
        m2 = self.__new_data_array.shape[1]
        index = [[],[]]
        error = 0

        for i in range(0,m1,int(m1/200.0)):
            dx = self.__new_data_array - np.matlib.repmat( self.transposition( self.__pre_data_array[:,i] ), 1, m2 )
            dist = np.sqrt(dx[0,:] ** 2 + dx[1,:] ** 2)

            ii = np.argmin(dist)
            dist = np.min(dist)

            #勾配を考慮
            if len(self.__pre_gradientList) != 0 and self.__pre_gradientList[i] > 0 and self.__now_gradientList[ii] > 0:
                dist *= abs(self.__pre_gradientList[i] - self.__now_gradientList[ii])

            #点の端(境界)を無視および一定以上距離の離れている点を除去
            if (len(self.__pre_boundaryList) != 0 and ii in self.__now_boundaryList) or dist > 0.1:
                continue

            index[0].append(i)
            index[1].append(ii)
            error = error + dist

        return index , error

    #特異値分解法による並進ベクトルと、回転行列の計算
    def SVDMotionEstimation(self, index):
        #print("data size:{}=>{}".format(len(data1[0]),len(index[0])))
        #各点群の重心の計算
        M = self.__pre_data_array[:,index[0]]
        mm = np.c_[M.mean(1)]
        S = self.__new_data_array[:,index[1]]
        ms = np.c_[S.mean(1)]

        #各点群を重心中心の座標系に変換
        Sshifted = np.array([S[0,:] - ms[0],
                             S[1,:] - ms[1]])
        Mshifted = np.array([M[0,:] - mm[0],
                             M[1,:] - mm[1]])

        W = Sshifted.dot(self.transposition( Mshifted ))
        U,A,V = np.linalg.svd( W )    #特異値分解

        R = self.transposition( U.dot( self.transposition( V ) ))   #回転行列の計算
        t = mm - R.dot(ms) #並進ベクトルの計算

        return R , t

    # degree to radian
    def toRadian(self , degree):
        return degree / 180 * pi

    # arrayを転置する
    # numpy標準では1行(ベクトル)だと転置されないので...
    def transposition(self , array):
        if len(array.shape) == 1 or array.shape[0] == 1:
            return np.c_[array]
        else:
            return array.T
    #境界のリストを作成
    def make_boundary_list(self):
        self.__pre_boundaryList = self.__now_boundaryList

        boundaryFlag = False #ゴミ値がfalse
        datacount = 0
        for index, data in enumerate( self.__new_data_array):
            if float(data[0]) == 0.0 and float( data[1]) == 0.0:
                if boundaryFlag:
                    boundaryFlag = not boundaryFlag
                    self.__now_boundaryList.append(index)

                datacount = 0
                continue

            datacount += 1

            if not boundaryFlag:
                boundaryFlag = not boundaryFlag
                self.__now_boundaryList.append(index)

        if boundaryFlag:
            boundaryFlag = not boundaryFlag
            self.__now_boundaryList.append(len( self.__new_data_array ) - 1)

    #各点の勾配リストを作成
    def make_gradient_list(self):
        self.__pre_gradientList = self.__now_gradientList
        for index in range(len(self.__new_data_array[0])):
            if index in self.__now_boundaryList or index == 0 or index == len(self.__new_data_array[0])-1:
                self.__now_gradientList.append(-1.0)
            else:
                self.__now_gradientList.append(
                    self.calcGradient(
                    [self.__new_data_array[0][index] - self.__new_data_array[0][index - 1],
                     self.__new_data_array[1][index] - self.__new_data_array[1][index - 1]],
                    [self.__new_data_array[0][index + 1] - self.__new_data_array[0][index],
                     self.__new_data_array[1][index + 1] - self.__new_data_array[1][index]]
                    ))

    #勾配の計算
    #勾配っていうか三点のなす角？
    def calcGradient(self, vector1,vector2):
        det = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        inner = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        return atan2(det,inner)