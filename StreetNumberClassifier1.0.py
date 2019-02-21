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
K.tensorflow_backend._get_available_gpus()
from matplotlib.pyplot import imshow
import numpy as np

import scipy.io
import matplotlib.pyplot as plt
from numpy import random
from numpy import zeros

#helper function for generateGray using the formula from internet
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#convert coloforful image to grayscale image (reduce computation later)
def generateGray(X):
    ans=zeros([X.shape[3],32,32])
    for i in range(X.shape[3]):
        ans[i]=rgb2gray(X[:,:,:,i])  
    return ans


#load our dataset
train_data = scipy.io.loadmat('train_32x32.mat')
# extract the images and labels from the dictionary object
X = train_data['X']
print(X.shape)
y = train_data['y']
# view an image (e.g. 25) and print its corresponding label
img_index = 1123
plt.imshow(X[:,:,:,img_index])
plt.show()
# gray shape: 73257x32x32
gray=generateGray(X)
print(gray.shape) 

#i do not understand why i must add plt.get_cmap('gray') to make it looks like grayscale
#if plt.get_cmap('gray') is removed, you can still see colors.
imshow(gray[img_index],cmap = plt.get_cmap('gray'))

