#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Task: linear kalman filter for pedestrain (2 dimensional)
url : https://blog.csdn.net/qq_40429562/article/details/98170564
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, Matrix


x = np.Matrix([[0.0, 0.0, 0.0, 0.0]]).T
print(x, x.shape)

P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
print(P, P.shape)

dt = 0.1 # Time Step between Filter Steps
F = np.matrix([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
print(F, F.shape)

H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(H, H.shape)


ra = 0.0



