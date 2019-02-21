#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:56:02 2019

@author: larry1285
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from matplotlib.pyplot import imshow
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("before:",x_train.shape)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
print("after:",x_train.shape)
print(y_train[0])
#astype函数用于array中数值类型转换
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#print (y_train[0])

# convert class vectors to binary class matrices eg: 5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

#print(y_train[0])

model = Sequential()
model.add(Dense(20, activation=LeakyReLU(), input_shape=(784,)))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(20, activation=LeakyReLU()))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


batch_size = 128
num_classes = 10
epochs = 10
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))