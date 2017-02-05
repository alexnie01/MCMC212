# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:48:11 2016

@author: anie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from pprint import pprint

param_keys = {'H': 1,
              'ombh2':2,
              'omch2': 3,
              'tau': 4,
              'ns': 5,
              'As': 6}
#data = np.loadtxt('chain.dat')              
#data = np.loadtxt('chain_12_200.dat')
data = np.loadtxt('chain2.dat')       
burn_data = np.loadtxt('chain_12_50.dat')
#data = np.loadtxt('chain_single_16_400.dat')
#extend = np.loadtxt('chain_single_16_400_extend.dat')


       
nwalkers = 16
n_params = 6
burnout = 180

data = np.concatenate((burn_data[:burnout*nwalkers], data[burnout*nwalkers:]))
#data = np.concatenate((data, extend))

fig, axes = plt.subplots(2,3, figsize = (14,14))
for i in np.arange(nwalkers):
    for j in np.arange(n_params):
        axes[j/3, j%3].plot(data[i::12, j+1])
axes[0,0].set_title('$H_0$')
axes[0,1].set_title('$\Omega_b h^2$')
axes[0,2].set_title('$\Omega_c h^2$')
axes[1,0].set_title('$\\tau$')
axes[1,1].set_title('$n_s$')
axes[1,2].set_title('$A_s$')

plt.draw()
#fig.savefig('runs_16_800.png')
#fig.savefig('runs_12_200.png')
#fig.savefig('runs_12_700.png')

eq_data = data[burnout*nwalkers:, 1:]

stats = {}

for key in param_keys:
    stats[key + '_mean'] = data[:, param_keys[key]].mean()
    stats[key + '_stderr'] = data[:, param_keys[key]].std()

pprint(stats)

corner_fig = corner.corner(eq_data, 
                           labels = ["$H_0$", "$\Omega_b h^2$", "$\Omega_c h^2$",
                                              "$\\tau$", "$n_s$", "$A_s$"],
                           truths = [stats['H_mean'], stats['ombh2_mean'],
                                     stats['omch2_mean'], stats['tau_mean'],
                                     stats['ns_mean'], stats['As_mean']])
#corner_fig.savefig("data_16_800.png")
#corner_fig.savefig('data_12_200.png')
#corner_fig.savefig('data_12_700.png')
#
    