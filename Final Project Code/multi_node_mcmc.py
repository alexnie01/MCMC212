# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 02:03:30 2016

@author: anie
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from ensemble import EnsembleSampler
#import matplotlib.pyplot as plt
import numpy as np

import camb
from camb import model, initialpower
#import time
#import sys
#import corner
#import pdb

from emcee.utils import MPIPool

# read Planck file    
filename = 'TTHILUB.txt'                   
TT_ = np.loadtxt('TTHILUB.txt', skiprows = 3)
scale=7.4311e12

# emcee params
threads = 12
nsteps = 2
ndim, nwalkers = 6, 12

# calculate log likelihood for single Cl against data
def lnlkhood(row):
    Clt_ = row[0]
    ClT_ = row[1]
    return -np.log(ClT_[2]*np.sqrt(2*np.pi))-.5*((Clt_-ClT_[1])/ClT_[2])**2

# return likelihood from CAMB of a single set of cosmological parameters
def run_camb(TT_, H0=67.5, ombh2=.022, omch2=.122,tau=.07,ns=.965,As=2e-9,lmax=2500,
             scale=7.4311e12):
    # set up and run CAMB
    pars = camb.CAMBparams()
    # cosmological setup
    cosmology = {'H0': H0,
                 'ombh2' : ombh2,
                 'omch2': omch2,
                 'tau' : tau}
    params = {'ns': ns,
              'As': As}
              
    pars.set_cosmology(**cosmology)
    pars.InitPower.set_params(**params)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    
    # get camb results
    results = camb.get_results(pars)
    # get matter-power spectrum
    powers = results.get_cmb_power_spectra(pars)
    
    # total Cl's
    totCL = powers['total']
    # unlensed Cl's
    unlensedCL = powers['unlensed_scalar']

    # rescale to Planck data units

    tt_ = scale * totCL[:, 0]
    
    dats = zip(tt_[30:2500],TT_[:2470])
    return sum(map(lnlkhood,dats))

def run_camb_wrapper(theta, TT_):
    H0, ombh2, omch2, tau, ns, As = theta
    return run_camb(TT_, H0, ombh2, omch2, tau, ns, As)

def lnprior(theta):
    H0, ombh2, omch2, tau, ns, As = theta
    if (50 < H0 < 100 and .005 < ombh2 < .1 and .001 < omch2 < .5 and .01 < tau < .13
        and .8 < ns < 1.2 and 2 < np.log(1e10 * As) < 4):
        return -np.log(.02*np.sqrt(2*np.pi))-.5*((tau-.07)/.02)**2
    return -np.inf
    
def lnprob(theta, TT_):
    lp = lnprior(theta)
    if np.isinf(lp):
#        sys.stdout.write('null prior {}'.format(theta))
        return -np.inf
#    sys.stdout.write('accepted')
    return lp + run_camb_wrapper(theta, TT_)

if __name__ == '__main__':
#    t0 = time.time()
    H0=67.5
    ombh2=.022
    omch2=.122
    tau=.07
    ns=.965
    As=2e-9
    lmax=2500
    scale=7.4311e12
    
    pos = 2*np.random.rand(nwalkers, ndim)-1
    
    pos[:, 0] = H0 + 5*pos[:,0]
    pos[:, 1] = ombh2 + .005*pos[:, 1]
    pos[:, 2] = omch2 + .004*pos[:, 2]
    pos[:, 3] = tau + .02*pos[:, 3]
    pos[:, 4] = ns + .05*pos[:, 4]
    pos[:, 5] = As + 5e-10*pos[:, 5]
    
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    sampler = Sampler(nwalkers, ndim, lnprob, threads = threads, 
                                    args=(TT_,),pool=pool)

    pool.close()

    ##sampler.run_mcmc(pos, 2000)
    width = 30
    count = 0
    for result in sampler.sample(pos, iterations = nsteps, storechain = True):
        position = result[0]
        f = open('chain.dat', 'a')
        for k in np.arange(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k].astype('|S32'))))
        f.close()
        count += 1
        n = int((width+1) * float(count) / nsteps)
#        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
#    sys.stdout.write("\n")
#    t1 = time.time()
#    print "finished emcee"
#    fig = plt.figure(1)
#    ax = fig.add_subplot(111)
    
#    for i in np.arange(0,nwalkers):
#        position = sampler.chain[i]
#        H = position[:,0]
#        ax.plot(H)
        
    #axes = np.array([])
    #for i in np.arange(ndim):
    #    axes = np.append(axes,fig.add_subplot(311+i))
    #t2 = time.time()
    #print "plotting"
    #for j in np.arange(nwalkers):
    #    m_samp, b_samp, f_samp = sampler.chain[j].T
    #    axes[0].plot(m_samp)
    #    axes[1].plot(b_samp)
    #    axes[2].plot(f_samp)
    #t3 = time.time()
    #
    #samples = sampler.chain[:, 200:,].reshape((-1,ndim))
    #
    #fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
    #                    truths=[m_true, b_true, np.log(f_true)])
    #fig.savefig("line.png")
    #
    #samples[:, 2] = np.exp(samples[:, 2])
    #m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #                             zip(*np.percentile(samples, [16, 50, 84],
    #                                                axis=0)))
    #print (m_mcmc, b_mcmc, f_mcmc)                                        
    #
    #print "plotting time: {}".format(t3-t2)
    
    
    
    
    
    
    
    
