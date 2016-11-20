# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:24:34 2016

@author: anie
"""

import numpy as np
import emcee as mc

# return negative log likelihood of multivariate gaussian

def lnP(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov, diff))/2.0
    
# number of hyperparameters
ndim = 50
# randomize 50 dimensional mean and covariances
means = np.random.rand(ndim)
cov = .5 - np.random.rand(ndim ** 2).reshape((ndim,ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)