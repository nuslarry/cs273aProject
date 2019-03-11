#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:43:07 2019

@author: larry1285
"""
import keras
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
(X_train, y_train), (X_test,y_test) = mnist.load_data()
print(X_train.shape)
print("hi")
X_train = X_train.reshape(-1, 1,28, 28)
X_test = X_train.reshape(-1, 1,28, 28)
y_train=np_utils.to_categorical(y_train, num_classes=10)
y_test=np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape)
 
model = Sequential()
#Conv layer 1 output shape(32, 28, 28)
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',  #padding method
    input_shape=(1, # channels
                 28,28), #height & width
              
))
model.add(Activation('relu'))

#Pooling layer 1 (max pooling) output shape(32,14,14)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same', # padding method
        ))

#Con layer 2 output shape(64,14,14)
model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))
#poolint layer 2 (max pooling ) output shape(64,7,7)
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

#fullt connected layer 1 input shape(64 * 7 * 7) = 3136, output shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#fullt connected layer 2 to shape(10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

#another way to define your optimizer
adam=Adam(lr=1e-4)

#we add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=1, batch_size=32)
loss, accuracy= model.evaluate(X_test,y_test)
print("test loss:",loss)
print("test loss:",accuracy)




