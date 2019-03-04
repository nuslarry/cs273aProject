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
import utilities


#load our dataset
train_data = scipy.io.loadmat('train_32x32.mat')
test_data  = scipy.io.loadmat('test_32x32.mat')

#x_train : 73257x1024 , y_train : 73257 x 11
x_train, y_train = utilities.preprocess(train_data)
x_test, y_test =  utilities.preprocess(test_data)

imshow(x_test[4].reshape(32,32))
plt.show()
print (y_test[4])

#training
model=Sequential()
model.add(Dense(input_dim=32*32,units=633,activation='relu'))
#training set的performance不大好,所以先不加dropout
#model.add(Dropout(0.5))
model.add(Dense(units=633,activation='relu'))
model.add(Dense(units=633,activation='relu'))
model.add(Dense(units=633,activation='sigmoid'))
model.add(Dense(units=633,activation='sigmoid'))
model.add(Dense(units=10,activation='softmax'))
sgd = SGD(lr=0.1)

#用sgd,sigmoid performance很差,  換成relu,adam後train的速度有變快
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=500,epochs=9)
trainingSetresult =model.evaluate(x_train,y_train)
testingSetresult =model.evaluate(x_test,y_test)
print(trainingSetresult)
print(testingSetresult)
#print("Test accuracy: ", result[1])












