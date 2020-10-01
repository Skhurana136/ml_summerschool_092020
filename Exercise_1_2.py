# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:35:14 2020

@author: khurana
"""
import numpy as np

X1 = np.array([0.56, 0.22, 1.7, 0.63, 0.36, 1.2])
X2 = np.array([1,1,1,1,1,1])
X = np.array([X2,X1]).T
y = np.array([160, 150, 175, 185, 165, 170])

#Assume the following equation
# y = beta[0] + beta[1]x
#beta is the product of (inverse of (X.T and X), X.T and y)
beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),y)

#Assume quadratic equation
# y = beta[0] + beta[1]x + beta[2]x**2
X1 = np.array([0.56, 0.22, 1.7, 0.63, 0.36, 1.2])
X1sq = X1**2
X2 = np.array([1,1,1,1,1,1])
X = np.array([X2,X1, X1sq]).T
y = np.array([160, 150, 175, 185, 165, 170])
beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),y)