# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:35:14 2020

@author: khurana
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor


column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]

data = pd.read_csv("Y:/Home/khurana/1.Scripts/ml_summerschool_092020/datasets/abalone.data", names=column_names)

print("Number of samples: %d" % len(data))
data.head()

y = data[["rings"]]
X = data.drop(["rings"], axis = 1, inplace = False)

plt.scatter(data["length"], data["shell weight"], c = data["rings"])
plt.xlabel ("Length")
plt.ylabel ("Shell Weight")
plt.title ("Variation of age with length and weight of the shell")

def linreg_singfeature (data, target, features, ratio):
    X_temp = data[features]
#    y_temp = list(target.values)
    y_temp = target
    # Split the data into training/testing sets
    X_train = X_temp[:-ratio]
    X_test = X_temp[-ratio:]
    
    # Split the targets into training/testing sets
    y_train = y_temp[:-ratio]
    y_test = y_temp[-ratio:]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((regr.predict(X_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, y_test))
    X_test.sort_values(by = ["length", "shell weight"], inplace = True)

    y_pred = regr.predict(X_test)
   
    return y_pred, y_test

predictions, truth = linreg_singfeature (X, y, ["length", "shell weight"], 210)

#Compare prediction and truth
compy = list(range(int(np.round(min([np.min(predictions), truth.values.min()]),0))-1, int(np.round(max([np.max(predictions), truth.values.max()]),0))+1))
plt.scatter(predictions, truth)
plt.plot(compy, compy, c = "red")
plt.ylabel ("Truth")
plt.xlabel ("Prediction")


#K nearest neighbors

def run_knn (data, target, features, neardatapoints, weight, ratio):
    neigh = KNeighborsRegressor(n_neighbors=neardatapoints, weights = weight)
    X_temp = data[features]
    y_temp = target
    X_train = X_temp[:-ratio]
    X_test = X_temp[-ratio:]
    
    # Split the targets into training/testing sets
    y_train = y_temp[:-ratio]
    y_test = y_temp[-ratio:]
    
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test.sort_values(by = [features[0]]))
    
    #Plotting
#    plt.figure()
#    plt.title ("k-Nearest neighbors classification using distance weight and: " + str(neardatapoints))
#    plt.scatter(X_train[features[0]], y_train, s = 30, marker = "^", c = "grey", alpha = 0.2, label = "Training")
#    plt.scatter(X_test[features[0]], y_pred, s = 20, c='blue', label = "Prediction")
#    plt.scatter(X_test[features[0]], y_test, s = 30, c = "black", label = "Test")
#    plt.xlabel (features[0])
#    plt.ylabel ("Target")
#    plt.legend()
#    plt.show()
    
    return y_pred, y_test

for n in [7, 100, 1000]:
    predictions, truth = run_knn(X, y, ["length", "shell weight"], n, "distance", 210)
    compy = list(range(int(np.round(min([np.min(predictions), truth.values.min()]),0))-1, int(np.round(max([np.max(predictions), truth.values.max()]),0))+1))
    plt.figure()
    plt.scatter(predictions, truth)
    plt.plot(compy, compy, c = "red")
    plt.ylabel ("Truth")
    plt.xlabel ("Prediction")

