#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:34:54 2019

@author: larry1285
"""
#Progression Log
#2/20 - load data, understanding input structure, print image, , grayscale conversion 

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
K.tensorflow_backend._get_available_gpus()
from matplotlib.pyplot import imshow
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy import random
from numpy import zeros
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import utilities


#load our dataset
train_data = scipy.io.loadmat('train_32x32.mat')
test_data  = scipy.io.loadmat('test_32x32.mat')

#preprocessing
#x_train : 73257x1024 , y_train : 73257 x 11
x_train, y_train = utilities.preprocess(train_data)
x_test, y_test =  utilities.preprocess(test_data)
#imshow(x_test[4].reshape(32,32))
#plt.show()
#print (y_test[4])

#PCA
cov_mat = np.cov(x_train.T)
print(cov_mat.shape)# 输出為(1024, 1024）

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vecs.shape) # 输出為(1024, 1024）

tot = sum(eigen_vals) # 求出eigenvalues
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)] # 求出eigenvalues的值（降序排列）
cum_var_exp = np.cumsum(var_exp) # 返回var_exp的累積和

print(eigen_vals.shape)
print(cum_var_exp.shape)

var_exp=var_exp[:90]
eigen_vals=eigen_vals[:90]
cum_var_exp=cum_var_exp[:90]
print()
plt.bar(range(len(eigen_vals)), var_exp, width=1.0, bottom=0.0, alpha=0.5, label='individual explained variance')
plt.step(range(len(eigen_vals)), cum_var_exp, where='post', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()


eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))] # 把eigenvalue,eigenvector湊成一對
eigen_pairs.sort(reverse=True) # sort by eigenvalue

eigenVectorMatrix=np.array(eigen_pairs[0][1])[:,np.newaxis]
for i in range(1,80):
    eigenVectorMatrix=np.hstack((eigenVectorMatrix,eigen_pairs[i][1][:,np.newaxis]))
print(eigenVectorMatrix.shape)    
x_train_pca = x_train.dot(eigenVectorMatrix)
x_test_pca = x_test.dot(eigenVectorMatrix)

y_train_pca=y_train
y_test_pca=y_test
print(x_test_pca.shape)
print(type(x_test_pca))


adict = {}
adict['X'] = x_train_pca
adict['y'] = y_train_pca # exactly equaly to y_train
sio.savemat('train_80_pca.mat', adict)


adict = {}
adict['X'] = x_test_pca
adict['y'] = y_test_pca # exactly equaly to y_train
sio.savemat('test_80_pca.mat', adict)



# load data
# bdict = sio.loadmat('train_80_pca.mat')
# x_train_prime=bdict['x']
# print(x_train_prime.shape)





    
#first = eigen_pairs[0][1]
#second = eigen_pairs[1][1]
#print("first:",first.shape)
#first = first[:,np.newaxis]
#second = second[:,np.newaxis]
#w = np.hstack((first,second))

#x_train_pca = x_train.dot(w) # dimension reduction based on previous w


#convert data to 2d and plot it ( graph still too complex)
#colors = ['darkgreen','b', 'g', 'r', 'c', 'm', 'y', 'k', 'cyan','plum','olive']
#markers = ['4','s', 'x', 'o','p','P','+','D','d','>','<']
#for i in range(x_train_pca.shape[0]):
#    print("i=",i)
#    yRealValue=0
#    for j in range(len(y_train[i])):
#        if y_train[i][j]==1:
#           yRealValue=j
#           break
#    yRealValue=(yRealValue+1)%10
#    plt.scatter(x_train_pca[i, 0], x_train_pca[i, 1], c=colors[yRealValue], label=yRealValue, marker=markers[yRealValue])
#    
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#plt.show()