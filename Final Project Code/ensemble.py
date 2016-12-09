#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
"""

MCMC Solver based on "Stretch-Move Optimizer by Mackey 2013
"""

__all__ = ["EnsembleSampler"]

import numpy as np

import autocorr
from MCMCSolver import Sampler
from interruptible_pool import InterruptiblePool


class EnsembleSampler(Sampler):
    def __init__(self, nwalkers, dim, lnpostfn, a=2.0, args=[], kwargs={},
                 postargs=None, threads=1, pool=None,
                 runtime_sortingfn=None):
        self.k = nwalkers
        self.a = a
        self.threads = threads
        self.pool = pool
        self.runtime_sortingfn = runtime_sortingfn

        if postargs is not None:
            args = postargs
        super(EnsembleSampler, self).__init__(dim, lnpostfn, args=args,
                                              kwargs=kwargs)

        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.
        self.lnprob = _function_wrapper(self.lnprob, self.args,
                                          self.kwargs)

        assert self.k % 2 == 0, "Must have even number of walkers for pooling."
        assert self.k >= 2 * self.dim, (
                "Walker count too low. Must have 2N for N param exploration")
        # parallel processing with auto-threading. Better to multiprocess?            
        if self.threads > 1 and self.pool is None:
            self.pool = InterruptiblePool(self.threads)

    def clear_blobs(self):
        """
        Clear the ``blobs`` list.

        """
        self._blobs = []

    def reset(self):
        """
        Clear params for next step

        """
        # reset sampler
        super(EnsembleSampler, self).reset()
        self.naccepted = np.zeros(self.k)
        self._chain = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

        # Initialize list for storing optional metadata blobs.
        self.clear_blobs()

    def sample(self, pos0, lnprob0=None, rstate0=None, blobs0=None,
               iterations=1, thin=1, storechain=True, mh_proposal=None):
        """
        Advance the chain ``iterations`` steps as a generator.

        :param pos0:
            A list of the initial positions of the walkers in the
            parameter space. It should have the shape ``(nwalkers, dim)``.

        :param lnprob0: (optional)
            The list of log posterior probabilities for the walkers at
            positions given by ``pos0``. If ``lnprob is None``, the initial
            values are calculated. It should have the shape ``(k, dim)``.

        :param rstate0: (optional)
            The state of the random number generator.
            See the :attr:`Sampler.random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.

        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.

        :param mh_proposal: (optional)
            A function that returns a list of positions for ``nwalkers``
            walkers given a current list of positions of the same size. See
            :class:`utils.MH_proposal_axisaligned` for an example.

        At each iteration, this generator yields:

        * ``pos`` - A list of the current positions of the walkers in the
          parameter space. The shape of this object will be
          ``(nwalkers, dim)``.

        * ``lnprob`` - The list of log posterior probabilities for the
          walkers at positions given by ``pos`` . The shape of this object
          is ``(nwalkers, dim)``.

        * ``rstate`` - The current state of the random number generator.

        * ``blobs`` - (optional) The metadata "blobs" associated with the
          current position. The value is only returned if ``lnpostfn``
          returns blobs too.

        """
        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        self.random_state = rstate0

        p = np.array(pos0)
        halfk = int(self.k / 2)

        # If the initial log-probabilities were not provided, calculate them
        # now.
        lnprob = lnprob0
        blobs = blobs0
        if lnprob is None:
            lnprob, blobs = self._get_lnprob(p)

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(lnprob)):
            raise ValueError("The initial lnprob was NaN.")

        # Store the initial size of the stored chain.
        i0 = self._chain.shape[1]

        # Here, we resize chain in advance for performance. This actually
        # makes a pretty big difference.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                                          np.zeros((self.k, N, self.dim))),
                                         axis=1)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros((self.k, N))), axis=1)

        for i in range(int(iterations)):
            self.iterations += 1

            # If we were passed a Metropolis-Hastings proposal
            # function, use it.
            if mh_proposal is not None:
                # Draw proposed positions & evaluate lnprob there
                q = mh_proposal(p)
                newlnp, blob = self._get_lnprob(q)

                # Accept if newlnp is better; and ...
                acc = (newlnp > lnprob)

                # ... sometimes accept for steps that got worse
                worse = np.flatnonzero(~acc)
                acc[worse] = ((newlnp[worse] - lnprob[worse]) >
                              np.log(self._random.rand(len(worse))))
                del worse

                # Update the accepted walkers.
                lnprob[acc] = newlnp[acc]
                p[acc] = q[acc]
                self.naccepted[acc] += 1

                if blob is not None:
                    assert blobs is not None, (
                        "If you start sampling with a given lnprob, you also "
                        "need to provide the current list of blobs at that "
                        "position.")
                    ind = np.arange(self.k)[acc]
                    for j in ind:
                        blobs[j] = blob[j]

            else:
                # Loop over the two ensembles, calculating the proposed
                # positions.

                # Slices for the first and second halves
                first, second = slice(halfk), slice(halfk, self.k)
                for S0, S1 in [(first, second), (second, first)]:
                    q, newlnp, acc, blob = self._propose_stretch(p[S0], p[S1],
                                                                 lnprob[S0])
                    if np.any(acc):
                        # Update the positions, log probabilities and
                        # acceptance counts.
                        lnprob[S0][acc] = newlnp[acc]
                        p[S0][acc] = q[acc]
                        self.naccepted[S0][acc] += 1

                        if blob is not None:
                            assert blobs is not None, (
                                "If you start sampling with a given lnprob, "
                                "you also need to provide the current list of "
                                "blobs at that position.")
                            ind = np.arange(len(acc))[acc]
                            indfull = np.arange(self.k)[S0][acc]
                            for j in range(len(ind)):
                                blobs[indfull[j]] = blob[ind[j]]

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[:, ind, :] = p
                self._lnprob[:, ind] = lnprob
                if blobs is not None:
                    self._blobs.append(list(blobs))

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            if blobs is not None:
                # This is a bit of a hack to keep things backwards compatible.
                yield p, lnprob, self.random_state, blobs
            else:
                yield p, lnprob, self.random_state

    def _propose_stretch(self, pos0, pos1, lnprob0):
        """
        Generate next configuration to sample based on relative positions of
        walkers in each pool (pos0, pos1)

        :param lnprob0:
            The log-probabilities at ``pos0``.
        """
        s = np.atleast_2d(pos0)
        Ns = len(s)
        c = np.atleast_2d(pos1)
        Nc = len(c)

        # Generate z fraction path
        zz = ((self.a - 1.) * self._random.rand(Ns) + 1) ** 2. / self.a
        rint = self._random.randint(Nc, size=(Ns,))

        # Check lnprob of new configuration
        q = c[rint] - zz[:, np.newaxis] * (c[rint] - s)
        newlnprob, blob = self._get_lnprob(q)

        # Accept or reject new proposals
        lnpdiff = (self.dim - 1.) * np.log(zz) + newlnprob - lnprob0
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))

        return q, newlnprob, accept, blob

    def _get_lnprob(self, pos=None):
        if pos is None:
            p = self.pos
        else:
            p = pos

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("Unphysical configuration.")
        if np.any(np.isnan(p)):
            raise ValueError("Input error.")
            
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map

        # Calculate lnprob
        results = list(M(self.lnprob, [p[i] for i in range(len(p))]))

        try:
            lnprob = np.array([float(l[0]) for l in results])
            blob = [l[1] for l in results]
        except (IndexError, TypeError):
            lnprob = np.array([float(l) for l in results])
            blob = None

        # Check for lnprob returning NaN.
        if np.any(np.isnan(lnprob)):
            # Print some debugging stuff.
            print("NaN value of lnprob for parameters: ")
            for pars in p[np.isnan(lnprob)]:
                print(pars)

            # Finally raise exception.
            raise ValueError("lnprob returned NaN.")

        return lnprob, blob

    @property
    def blobs(self):
        return self._blobs

    @property
    def chain(self):
        """
        Pointer to MC data

        """
        return super(EnsembleSampler, self).chain

    @property
    def flatchain(self):
        """
        A shortcut for accessing chain flattened along the zeroth (walker)
        axis.

        """
        s = self.chain.shape
        return self.chain.reshape(s[0] * s[1], s[2])

    @property
    def lnprobability(self):
        """
        A pointer to the matrix of the value of ``lnprob`` produced at each
        step for each walker. The shape is ``(k, iterations)``.

        """
        return super(EnsembleSampler, self).lnprobability

    @property
    def flatlnprobability(self):
        """
        A shortcut to return the equivalent of ``lnprobability`` but aligned
        to ``flatchain`` rather than ``chain``.

        """
        return super(EnsembleSampler, self).lnprobability.flatten()

    @property
    def acceptance_fraction(self):
        """
        An array (length: ``k``) of the fraction of steps accepted for each
        walker.

        """
        return super(EnsembleSampler, self).acceptance_fraction

    @property
    def acor(self):
        """
        An estimate of the autocorrelation time for each parameter (length:
        ``dim``).

        """
        return self.get_autocorr_time()

    def get_autocorr_time(self, low=10, high=None, step=1, c=10, fast=False):
        """
        Check if system has reached equilibrium with autocorrelation of results
        """
        return autocorr.integrated_time(np.mean(self.chain, axis=0), axis=0,
                                        low=low, high=high, step=step, c=c,
                                        fast=fast)


class _function_wrapper(object):
    """
    Pickle functions for MPI

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
