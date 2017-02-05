# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:14:52 2016

@author: anie
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import camb

class Sampler2:
    def __init__(self, params, priors):
        self.params = params
        self.priors = priors
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

__all__ = ["Sampler"]

import numpy as np


class Sampler(object):
    def __init__(self, dim, lnprob, args=[], kwargs={}):
        self.dim = dim
        self.lnprob = lnprob
        self.args = args
        self.kwargs = kwargs

        self._random = np.random.mtrand.RandomState()

        self.reset()

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter  # NOQA
    def random_state(self, state):
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def acceptance_fraction(self):
        return self.naccepted / self.iterations

    @property
    def chain(self):
        """
        Pointer

        """
        return self._chain

    @property
    def flatchain(self):
        """
        Alias of ``chain`` provided for compatibility.

        """
        return self._chain

    @property
    def lnprobability(self):
        """
        Lnprob values of each step of chain

        """
        return self._lnprob

    @property
    def acor(self):
        return self.get_autocorr_time()

    def get_autocorr_time(self, window=50):
        raise NotImplementedError("Function not implemented")

    def get_lnprob(self, p):
        """Return the lnprob at the given position."""
        return self.lnprob(p, *self.args, **self.kwargs)

    def reset(self):
        """
        Clear ``chain``, ``lnprobability`` params
        """
        self.iterations = 0
        self.naccepted = 0
        self._last_run_mcmc_result = None

    def clear_chain(self):
        return self.reset()

    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sampling routine must be implemented "
                                  "by subclasses")

    def run_mcmc(self, pos0, N, rstate0=None, lnprob0=None, **kwargs):
        if pos0 is None:
            if self._last_run_mcmc_result is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            pos0 = self._last_run_mcmc_result[0]
            if lnprob0 is None:
                rstate0 = self._last_run_mcmc_result[1]
            if rstate0 is None:
                rstate0 = self._last_run_mcmc_result[2]

        for results in self.sample(pos0, lnprob0, rstate0, iterations=N,
                                   **kwargs):
            pass

        # store so that the ``pos0=None`` case will work.  We throw out the blob
        # if it's there because we don't need it
        self._last_run_mcmc_result = results[:3]

        return results

        

