# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 01:24:10 2016

@author: anie
"""
import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
import emcee
import time
import corner

# write to file
f = open('chain.dat', 'w')
f.close()

# actual parameters

m_true = -.423
b_true = 4.29
f_true = .542

# emcee params
threads = 4
nsteps = 2000
ndim, nwalkers = 3,20
pos = np.random.rand(nwalkers, ndim)
pos[:, 1] = 10 * pos[:, 1]

# Generate data

N = 50
x = np.sort(10*np.random.rand(N))
yerr = .1+.5*np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

fig = plt.figure(1)
ax = fig.add_subplot(111)
x_true = np.arange(0,10,.01)
y_true = m_true * x_true + b_true
ax.errorbar(x,y,yerr, fmt='o')
ax.plot(x_true, y_true)

# calculate likelihood of data given prior theta
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# return prior probability
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < .5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# return conditional probability
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
print "starting emcee"
t0 = time.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads = threads, args=(x,y,yerr))
#sampler.run_mcmc(pos, 2000)
width = 30
count = 0
for result in sampler.sample(pos, iterations = nsteps, storechain = True):
    position = result[0]
    f = open('line_chain.dat', 'a')
    for k in np.arange(position.shape[0]):
        f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k].astype('|S32'))))
    f.close()
    count += 1
    n = int((width+1) * float(count) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
sys.stdout.write("\n")
t1 = time.time()
print "finished emcee"
fig = plt.figure(2)
axes = np.array([])
for i in np.arange(ndim):
    axes = np.append(axes,fig.add_subplot(311+i))
t2 = time.time()
print "plotting"
for j in np.arange(nwalkers):
    m_samp, b_samp, f_samp = sampler.chain[j].T
    axes[0].plot(m_samp)
    axes[1].plot(b_samp)
    axes[2].plot(f_samp)
t3 = time.time()

samples = sampler.chain[:, 200:,].reshape((-1,ndim))

fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                    truths=[m_true, b_true, np.log(f_true)])
fig.savefig("line.png")

samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print (m_mcmc, b_mcmc, f_mcmc)                                        

print "emcee time: {}".format(t1-t0)
print "plotting time: {}".format(t3-t2)

import emcee

class Samp(emcee.EnsembleSampler):
    def compress(self,f,x):
        return map(f,x)