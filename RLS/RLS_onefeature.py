# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:47:32 2021

@author: Digvijay
"""

import numpy as np 

# Create data
X = np.random.rand(100,1)
y = 4 + 3*X +np.random.randn(100,1)

# concatenate for w0
Xwb = np.c_[np.ones((100,1)), X]

# Compute w directly
W = Xwb.T.dot(Xwb)
W = np.linalg.inv(W)
W = (W.dot(Xwb.T)).dot(y)

print(W)
