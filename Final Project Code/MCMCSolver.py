# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:14:52 2016

@author: anie
"""
from data_test import Samp
import numpy as np
import camb

class Sampler(Samp):
    # calculate log likelihood for single Cl against data
    def lnlkhood(self,row):
        return row
    # return likelihood from CAMB of a single set of cosmological parameters
    def run_camb(self,TT_, H0=67.5, ombh2=.022, omch2=.122,tau=.07,ns=.965,As=2e-9,lmax=2500,
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
        return sum(map(self.lnlkhood,dats))

    def run_camb_wrapper(self,theta, TT_):
        H0, ombh2, omch2, tau, ns, As = theta
        return self.run_camb(TT_, H0, ombh2, omch2, tau, ns, As)
    
    def lnprior(theta):
        H0, ombh2, omch2, tau, ns, As = theta
        if (50 < H0 < 100 and .005 < ombh2 < .1 and .001 < omch2 < .5 and .01 < tau < .13
            and .8 < ns < 1.2 and 2 < np.log(1e10 * As) < 4):
            return -np.log(.02*np.sqrt(2*np.pi))-.5*((tau-.07)/.02)**2
        return -np.inf
        
    def lnprob(self,theta, TT_):
        lp = self.lnprior(theta)
        if np.isinf(lp):
    #        sys.stdout.write('null prior {}'.format(theta))
            return -np.inf
    #    sys.stdout.write('accepted')
        return lp + self.run_camb_wrapper(theta, TT_)
        
        

