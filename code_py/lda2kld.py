#!/usr/bin/env python

""" """

__author__ = 'kln-courses'

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/home/kln')
docmat = np.genfromtxt('data/theta.csv', delimiter=',')

# KL divergence
def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0,(p-q) * np.log10(p / q), 0))
# smooth
def movavg(vect, n = 5) :
    ret = np.cumsum(vect, dtype = float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# kld for doc_n to mean doc_1:doc_n-1 
kld = np.zeros(len(docmat))    
for i in range(1,len(docmat)):
    submat = docmat[0:i,]
    tmp = np.zeros(len(submat))
    for ii in range(len(submat)):
        tmp[ii] = kl(submat[ii,],docmat[i,])
    kld[i] = np.mean(tmp)    
kld_smooth = movavg(kld-np.mean(kld), n=100)


plt.figure(1)
h = plt.plot(kld_smooth)
plt.setp(h, 'color', 'r', 'linewidth', 2.0)
plt.xlabel('Time Index')
plt.ylabel('Bits')
plt.axis([0, len(kld_smooth),min(kld_smooth)*1.1,max((kld_smooth))*1.1])
plt.grid(True)
plt.show()