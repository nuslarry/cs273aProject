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
from tempfile import TemporaryFile


#helper function for generateGray using the formula from internet
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#convert coloforful image to grayscale image (reduce computation later)
def generateGray(X):
    ans=zeros([X.shape[3],32,32])
    for i in range(X.shape[3]):
        ans[i]=rgb2gray(X[:,:,:,i])  
    return ans


def preprocess(data):
    x = data['X']
    # extract the images and labels from the dictionary object
    # X[0] : 32 ,number of pixels in x direction
    # X[1] : 32 ,number of pixels in y direction
    # X[2] : 3  RGB
    # X[3] : total input eg: 73257

    xPixels=x.shape[0]
    yPixels=x.shape[1]
    colors=x.shape[2]
    inputCount=x.shape[3]
    
    
    y = data['y']
    # view an image (e.g. 25) and print its corresponding label
    img_index = 1123
    #plt.imshow(x[:,:,:,img_index])
    #plt.show()
    
    # x shape: 73257x32x32
    x=generateGray(x)
    
    #x shape   73257x(32x32) = 73257x1024
    x=x.reshape(inputCount,xPixels*yPixels)
    #print(X.shape)
    #i do not understand why i must add plt.get_cmap('gray') to make it looks like grayscale
    #if plt.get_cmap('gray') is removed, you can still see colors.
    #imshow(x[img_index].reshape(32,32),cmap = plt.get_cmap('gray'))
    #plt.show()
    #astype函数用于array中数值类型转换
    x = x.astype('float32')
    #數字越接近1,表示該pixel被塗得越深
    x /= 255
    #use 11 rather than 10 here since 0 is represented by 10 instead of 0
    # y: 73257 x 11
    y = keras.utils.to_categorical(y, num_classes=11) 
    print("before", y.shape)
    #remove first column
    y=y[:,1:]
    print(x.shape) 
    print(y.shape)
    return x, y






#load our dataset
b = np.loadtxt('transformed_training_data_x.txt')
print(b.shape)