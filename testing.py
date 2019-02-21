#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:47:03 2019

@author: larry1285
"""
import numpy as np
a = np.array([[1,2,3], [4,5,6]])
print (a)
a=np.reshape(a, 6)
print(a)

aa = np.zeros((10, 2))
b = aa.T
print(aa)
print(b)

test1=np.array([[1,2,3], [4,5,6]])
test1 = test1.reshape(3,2)
print(test1)