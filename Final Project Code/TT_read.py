# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:49:03 2016

@author: anie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


 filename = 'TTHILUB.txt'
 names = ['ELL', 'D_ELL', 'ERR'],
 dtype = {'ELL':np.int32, 'D_ELL': np.float64, 'ERR': np.float64}
                }

#filename = 'TTHILB.txt'
#names = ['ELL', 'LMIN', 'LMAX', 'D_ELL', 'ERR'],
#dtype = {'ELL':np.float64, 'LMIN': np.float64, 'LMAX': np.float64,
#                         'D_ELL': np.float64, 'ERR': np.float64}
                         
                         
header = 0
skiprows = 3
delim_whitespace = True
                         
TT_ = pd.read_table(filename, header = header, skiprows = skiprows, 
                    dtype = dtype, names = ['ELL', 'LMIN', 'LMAX', 'D_ELL', 'ERR'], 
                    delim_whitespace = delim_whitespace)

fig = plt.figure('Spectra')
ax = fig.add_subplot(111)
ax.plot(TT_['ELL'].values, TT_['D_ELL'].values)