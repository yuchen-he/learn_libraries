#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Task: linear kalman filter from udacity course
url : https://blog.csdn.net/dshgers/article/details/81190826
'''
import numbers
import numpy as np
import matplotlib.pyplot as plt  
import math

# 卡尔曼滤波器需要调用的矩阵类
class Matrix(object):

      # 构造矩阵
      def __init__(self, grid):
          self.g = np.array(grid)
          self.h = len(grid)
          self.w = len(grid[0])

      # 单位矩阵
      def identity(n):
          return Matrix(np.eye(n))

      # 矩阵的迹
      def trace(self):
          if not self.is_square():
              raise(ValueError, "Cannot calculate the trace of a non-square matrix.")
          else:
              return self.g.trace()
      # 逆矩阵
      def inverse(self):
          if not self.is_square():
              raise(ValueError, "Non-square Matrix does not have an inverse.")
          if self.h > 2:
              raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")
          if self.h == 1:
              m = Matrix([[1/self[0][0]]])
              return m
          if self.h == 2:
              try:
                  m = Matrix(np.matrix(self.g).I)
                  return m
              except np.linalg.linalg.LinAlgError as e:
                  print("Determinant shouldn't be zero.", e)

      # 转置矩阵
      def T(self):
          T = self.g.T
          return Matrix(T)
                  
      # 判断矩阵是否为方阵
      def is_square(self):
          return self.h == self.w

      # 通过[]访问
      def __getitem__(self,idx):
          return self.g[idx]

      # 打印矩阵的元素
      def __repr__(self):
          s = ""
          for row in self.g:
              s += " ".join(["{} ".format(x) for x in row])
              s += "\n"
          return s

      # 加法
      def __add__(self,other):
          if self.h != other.h or self.w != other.w:
              raise(ValueError, "Matrices can only be added if the dimensions are the same")
          else:
              return Matrix(self.g + other.g)

      # 相反数
      def __neg__(self):
          return Matrix(-self.g)

      #减法
      def __sub__(self, other):
          if self.h != other.h or self.w != other.w:
              raise(ValueError, "Matrices can only be subtracted if the dimensions are the same")
          else:
              return Matrix(self.g - other.g)


      # 矩阵乘法：两个矩阵相乘
      def __mul__(self, other):
          if self.w != other.h:
              raise(ValueError, "number of columns of the pre-matrix must equal the number of rows of the post-matrix")    
          return Matrix(np.dot(self.g, other.g))
                    
      # 标量乘法：变量乘以矩阵          
      def __rmul__(self, other):
          if isinstance(other, numbers.Number):
              return Matrix(other * self.g)


# 生成汽车行驶的真实数据
# 汽车从以初速度v0，加速度a行驶10秒钟，然后匀速行驶20秒
# x0:initial distance, m
# v0:initial velocity, m/s
# a:acceleration，m/s^2
# t1:加速行驶时间，s
# t2:匀速行驶时间，s
# dt:interval time, s
def generate_data(x0, v0, a, t1, t2, dt):
      a_current = a
      v_current = v0
      t_current = 0
      
      # 记录汽车运行的真实状态
      a_list = []
      v_list = []
      t_list = []

      # 汽车运行的两个阶段
      # 第一阶段：加速行驶
      while t_current <= t1:
          # 记录汽车运行的真实状态
          a_list.append(a_current)
          v_list.append(v_current)
          t_list.append(t_current)
          # 汽车行驶的运动模型
          v_current += a * dt
          t_current += dt

      # 第二阶段：匀速行驶
      a_current = 0
      while t2 > t_current >= t1:
          # 记录汽车运行的真实状态
          a_list.append(a_current)
          v_list.append(v_current)
          t_list.append(t_current)
          # 汽车行驶的运动模型
          t_current += dt

      # 计算汽车行驶的真实距离
      x = x0
      x_list = [x0]
      for i in range(len(t_list) - 1):
          tdelta = t_list[i+1] - t_list[i]
          x = x + v_list[i] * tdelta + 0.5 * a_list[i] * tdelta**2
          x_list.append(x)
      return t_list, x_list, v_list, a_list

# 生成雷达获得的数据。需要考虑误差，误差呈现高斯分布
def generate_lidar(x_list, standard_deviation):
      return x_list + np.random.normal(0, standard_deviation, len(x_list))

# 获取汽车行驶的真实状态
t_list, x_list, v_list, a_list = generate_data(100, 5, 4, 10, 20, 0.1)

# 创建激光雷达的测量数据
# 测量误差的标准差。为了方便观测，可以增加该值。
# standard_deviation = 0.15
standard_deviation = 0.3
# 雷达测量得到的距离
lidar_x_list = generate_lidar(x_list, standard_deviation)
#　雷达测量的时间
lidar_t_list = t_list


# 可视化.创建包含2*3个子图的视图
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 15))

# 真实距离
ax1.set_title("truth distance")
ax1.set_xlabel("time")
ax1.set_ylabel("distance")
ax1.set_xlim([0, 21])
ax1.set_ylim([0, 1000])
ax1.plot(t_list, x_list)

# 真实速度
ax2.set_title("truth velocity")
ax2.set_xlabel("time")
ax2.set_ylabel("velocity")
ax2.set_xlim([0, 21])
ax2.set_ylim([0, 50])
ax2.set_xticks(range(21))
ax2.set_yticks(range(0, 50, 5))
ax2.plot(t_list, v_list)

# 真实加速度
ax3.set_title("truth acceleration")
ax3.set_xlabel("time")
ax3.set_ylabel("acceleration")
ax3.set_xlim([0, 21])
ax3.set_ylim([0, 5])
ax3.plot(t_list, a_list)

# 激光雷达测量结果
ax4.set_title("Lidar measurements VS truth")
ax4.set_xlabel("time")
ax4.set_ylabel("distance")
ax4.set_xlim([0, 21])
ax4.set_ylim([0, 1000])
ax4.set_xticks(range(21))
ax4.set_yticks(range(0, 1000, 100))
ax4.plot(t_list, x_list, label="truth distance")
ax4.scatter(lidar_t_list, lidar_x_list, label="Lidar distance", color="red", marker="o", s=2)
ax4.legend()


# 使用卡尔曼滤波器
# 初始距离。注意：这里假设初始距离为0，因为无法测量初始距离。
initial_distance = 0

# 初始速度。注意：这里假设初始速度为0，因为无法测量初始速度。
initial_velocity = 0

# 状态矩阵的初始值
x_initial = Matrix([[initial_distance], [initial_velocity]])

# 误差协方差矩阵的初始值
P_initial = Matrix([[5, 0], [0, 5]])

# 加速度方差
acceleration_variance = 50

# 雷达测量结果方差
lidar_variance = standard_deviation**2

# 观测矩阵，联系预测向量和测量向量
H = Matrix([[1, 0]])

# 测量噪音协方差矩阵。因为测量值只有位置一个变量，所以这里是位置的方差。
R = Matrix([[lidar_variance]])

# 单位矩阵
I = Matrix.identity(2)

# 状态转移矩阵
def F_matrix(delta_t):
    return Matrix([[1, delta_t], [0, 1]])

# 外部噪音协方差矩阵
def Q_matrix(delta_t, variance):
    t4 = math.pow(delta_t, 4)
    t3 = math.pow(delta_t, 3)
    t2 = math.pow(delta_t, 2)
    return variance * Matrix([[(1/4)*t4, (1/2)*t3], [(1/2)*t3, t2]])

def B_matrix(delta_t):
    return Matrix([[delta_t**2 / 2], [delta_t]])

# 状态矩阵
x = x_initial

# 误差协方差矩阵
P = P_initial

# 记录卡尔曼滤波器计算得到的距离
x_result = []

# 记录卡尔曼滤波器的时间
time_result = []

# 记录卡尔曼滤波器得到的速度
v_result = []

for i in range(len(lidar_x_list) - 1):
    delta_t = (lidar_t_list[i + 1] - lidar_t_list[i]) 
    # 预测
    F = F_matrix(delta_t)
    Q = Q_matrix(delta_t, acceleration_variance)
    
    # 注意：运动模型使用的是匀速运动，汽车实际上有一段时间是加速运动的
    x_prime = F * x
    P_prime = F * P * F.T() + Q
    
    # 更新
    # 测量向量和状态向量的差值。注意：第一个时刻是没有测量值的，
    # 只有经过一个脉冲周期，才能获得测量值。
    y = Matrix([[lidar_x_list[i + 1]]]) - H * x_prime
    S = H * P_prime * H.T() + R
    K = P_prime * H.T() * S.inverse()
    x = x_prime + K * y
    P = (I - K * H) * P_prime
    x_result.append(x[0][0])
    v_result.append(x[1][0])
    time_result.append(lidar_t_list[i+1])

# 把真实距离、激光雷达测量的距离以及卡尔曼滤波器的结果（距离）可视化
ax5.set_title("Lidar measurements VS truth")
ax5.set_xlabel("time")
ax5.set_ylabel("distance")
ax5.set_xlim([0, 21])
ax5.set_ylim([0, 1000])
ax5.set_xticks(range(0, 21, 2))
ax5.set_yticks(range(0, 1000, 100))
ax5.plot(t_list, x_list, label="truth distance", color="blue", linewidth=1)
ax5.scatter(lidar_t_list, lidar_x_list, label="Lidar distance", color="red", marker="o", s=2)
ax5.scatter(time_result, x_result, label="kalman", color="green", marker="o", s=2)
ax5.legend()

# 把真实速度、卡尔曼滤波器的结果（速度）可视化
ax6.set_title("Lidar measurements VS truth")
ax6.set_xlabel("time")
ax6.set_ylabel("velocity")
ax6.set_xlim([0, 21])
ax6.set_ylim([0, 50])
ax6.set_xticks(range(0, 21, 2))
ax6.set_yticks(range(0, 50, 5))
ax6.plot(t_list, v_list, label="truth velocity", color="blue", linewidth=1)
ax6.scatter(time_result, v_result, label="Lidar velocity", color="red", marker="o", s=2)
ax6.legend()

plt.show()


'''

Another program for linear kalman filter
url: https://qiita.com/harmegiddo/items/ddd33f40d5e368a210df

'''

# import numpy as np
# import numpy.random as random
# import matplotlib.pyplot as plt

# # generate norm
# def norm(_loc=0.0, _scale=1.0, _size=(1)):
#     return random.normal(_loc, _scale, _size) 

# ####################################################################
# ###########################  初期値の設定  ##########################
# ####################################################################
# # 時刻
# global_time = 0
# # t+1の刻み設定
# dt=0.1
# # 計算回数
# calc_num = 50
# end_time = calc_num * dt 
# # 正規分布の発生に関するパラメータ
# mean_a = 0
# sigma_a = 1
# mean_z = 0
# sigma_z = 1
# # 単位行列
# I = np.matrix([[1 , 0],
#                [0, 1]])
# # Plot用
# ground_truth_position=[]
# observed_position=[]
# estimate_position=[]
# time_series=[]

# ##########       状態方程式        ##########
# # トロッコの位置と速度: [位置, 加速度]
# x_k = np.matrix([ 
#                 [0],
#                 [0]
#             ]) 

# x_k_true = x_k
# x_k_k = x_k
# # 運動方程式 [位置+(加速度*時間), 加速度]
# F = np.matrix([ 
#                 [1, dt],
#                 [0, 1]
#             ]) 
# # 時間遷移に関する雑音モデルの行列 (0平均かつQの正規分布に従う)
# G = np.matrix([
#                 [(dt**2) / 2],
#                 [dt]
#             ])

# ##########       観測方程式        ##########
# # 誤差行列
# p_k = np.matrix([
#                 [0, 0],
#                 [0, 0]
#              ])
# p_k_k = p_k
# # 位置のみを線形写像する観測モデル
# H = np.matrix([
#                 1,
#                 0
#              ])
# # cov(Gw_k) = (sigma_a)^2 * (G)(G^T): 共分散
# Q = (sigma_a**2) * G * G.T
# # R = E(v_k*(v_k)^t) = (sigma_z)^2: ?
# R = sigma_z**2


# ####################################################################
# ###########################  計算スタート  ##########################
# ####################################################################
# while global_time < end_time:

#     ##########       観測        ##########

#     # v_k: 観測誤差 (偶然誤差)
#     v_k = norm(mean_z, sigma_z)
#     # z_k = Hx_k + v_k: トロッコの位置をセンサで観測する
#     # z_k = H * x_k_true + v_k
#     z_k = H * x_k_true + v_k
#     observed_position.append(z_k.tolist()[0][0])

#     ##########       予測        ##########
#     # w_k = [a_k]: トロッコの加速度(誤差混み)
#     w_k = norm(mean_a, sigma_a)
#     # Fx_{k-1} + Gw_k: 現時刻における予測推定値
#     x_k = (F * x_k_k) + (G * w_k)

#     ## 真値(次の位置)
#     # Fx_{k-1} + Gw_k: （補正しない）現時刻における予測推定値
#     x_k_true = (F * x_k_true) + (G * w_k)
#     ground_truth_position.append(x_k_true.tolist()[0][0])

#     ##########     補正と更新     ##########
#     # F * P_{k-1} * F^T + G_k * Q_k * (G_k)^T: 現時刻における予測誤差行列
#     p_k = F * p_k_k * F.T + Q
#     # R + H * P_k * H^T: 観測残差の共分散
#     S_k = (H * p_k) * H.T + R
#     # P_k * H^T * S^-1: 最適カルマンゲイン
#     K_k = p_k * H.T * S_k.I
#     # z_k - H * x_k: 観測残差
#     e_k = z_k - H * x_k
#     # x_k + K_k * e_k: 位置の補正
#     x_k_k = x_k + K_k * e_k
#     estimate_position.append(x_k_k.tolist()[0][0])
#     # (I - K_k * H) * p_k_k: 更新された誤差の共分散
#     p_k_k = (I - K_k * H) * p_k



#     ##########    タイムカウント    ##########
#     time_series.append(global_time)
#     global_time += dt


# MSE = np.sum((np.array(ground_truth_position)-np.array(observed_position))**2)
# print("観測誤差"+str(MSE))
# MSE = np.sum((np.array(ground_truth_position)-np.array(estimate_position))**2)
# print("カルマンフィルタによる推定誤差"+str(MSE))

# plt.plot(time_series, ground_truth_position, color="blue", marker="", label="groundtruth")
# plt.plot(time_series, estimate_position, color="red", marker="+", label="estimation")
# plt.plot(time_series, observed_position, color="green", marker="", label="observed")
# plt.legend()
# plt.show()
