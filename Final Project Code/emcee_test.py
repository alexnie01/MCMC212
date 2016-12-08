# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:24:34 2016

@author: anie
"""

import numpy as np
import emcee
from pprint import pprint
import matplotlib.pyplot as plt

# return negative log likelihood of multivariate gaussian

def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov, diff))/2.0
    
# number of hyperparameters
ndim = 25
# randomize 50 dimensional mean and covariances
means = np.random.rand(ndim)
cov = .5 - np.random.rand(ndim ** 2).reshape((ndim,ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)
cov = np.triu(cov)
cov += cov.T-np.diag(cov.diagonal())
cov = np.dot(cov,cov)

icov = np.linalg.inv(cov)
print icov

# initialize positions of walkers
nwalkers = 100
p0 = np.random.rand(ndim *nwalkers).reshape((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])

# burn-in for walkers
pos, prob, state = sampler.run_mcmc(p0, 100)
# clear burn-in results
sampler.reset()

# run walkers and collect actual data
sampler.run_mcmc(pos, 1000)

# print rate at which new points were explored. Should be between .2 and .5
print ('Mean acceptance fraction: {0:.3f}'.format(np.mean(sampler.acceptance_fraction)))
# plot sampled density
for i in range(ndim):
    plt.figure()
    plt.hist(sampler.flatchain[:, i], 100, color = 'k', histtype = 'step')
    plt.title('Dimension {0:d}'.format(i))
    
plt.show()
