#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:41:55 2019

@author: larry1285
"""

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
y=np.array([[1,2,3],[4,5,6]])
print(y.shape)
y=y[:,(1,-1)]
print(y)