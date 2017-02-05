# MCMC212
n-parameter MCMC Solver for CMB Power Spectrum

This project implements a Markov Chain-Monte Carlo Solver to estimate cosmological parameters given higher moments (l>30) from the angular temperature correlation coefficients of the Cosmic Microwave Background's Power Spectrum. Marginalization of auxiliary variables is accomplished using the well-known Code for Anisotropies in Microwave Background (CAMB) package implemented by Lewis & Challinor. Two versions of the program are included: 

1. single_node_mcmc.py is designed to run on a small number of cores (usually a laptop's)
2. multi_node_mcmc.py is designed to run on clusters (ex: Harvard's ODYSSEY cluster)

Once the simulation is finished, the program outputs the maximum likelihood estimate of each of the parameters as well as a grid of joint probabilities between cosmological parameters. 

For more background on the algorithms and cosmology of the project, please refer to the included writeup (MCMC_Writeup.pdf).

Setup:

0. This package is includes files in both Python and Fortran, so a copy of Python 2.7 and the g++ compiler are needed.
1. Download and install the emcee (http://dan.iel.fm/emcee/current/) and CAMB packages (http://camb.info/readme.html)
2. Acquire CMB Power Spectrum data. Recent l-data from Plack's 2013 has been included, although you may supply your own.
3. Run single_node_mcmc.py or multi_node_mcmc.py depending on platform
4. Run Plotting.py to view results in cross-grid

Adjustable Parameters

single_node_mcmc.py and multi_node_mcmc.py

threads: number of threads to run. 
nsteps: number of steps to simulate. 
ndim: number of parameters to estimate
nwalkers: number of walkers in MCMC Solver. Must be at least 2*ndim
H0, ombh2, omch2, tau, ns, As, scale: maximum of prior for parameters to be estimated. Note these will be set by default if no values are supplied

Plotting.py

burnout: number of steps to ignore. It is important to have sufficiently high burnout in order to erase imprints from the initial conditions when calculating equilibrium constants



