# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:35:14 2020

@author: khurana
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[4, 4, 6, 7,  7, 8, 9, 9],
      [9, 6, 5, 3, 8, 5, 7, 8]]).T
y = np.array([1 , 2, 1, 2, 1, 2, 1, 2]).T

plt.scatter(X[:,0], X[:,1], c = y)

X0p = 7
X1p = 6

mandist =  