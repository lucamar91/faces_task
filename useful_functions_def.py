# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy.sparse.linalg import eigsh
from scipy.stats import unitary_group, ortho_group
from scipy.interpolate import interp1d
from datetime import datetime
import math
from sklearn.preprocessing import normalize

# THIS VERSION OF THE FILE IS COMPATIBLE W PYTHON3, WHICH IS NEEDED TO WORK ON MY MACBOOK :( - Luca
# basically the only differences are that print pippo --> print(pippo)   ;   is --> ==


def standardize(data):
    avg = np.average(data, axis=0)
    delta = data - avg                                # SxD array
    delta_std = delta/np.std(delta, axis=0, ddof=1)
    return delta_std


def correlation_matrix(data):
    delta_std = standardize(data)
    return np.cov(delta_std, rowvar=False)


def order(array, which='descending'):
    if which == 'descending':
        step = -1
    if which == 'ascending':
        step = 1        
    order = np.argsort(array)[::step]
    return np.copy(array)[order]


def mahalanobis_d(x1, x2, E, lambdas, P = None):
    # at least one of x1, x2 must be a vector (1D) or be both the same shape

    #per far funzionare sempre tutto devo convertire eventuali vettori (P,) in array (1,P)
    if len(x1.shape) == 1:
        x1 = np.array([x1])
    if len(x2.shape) == 1:
        x2 = np.array([x2])

    if P == None:
        P = x1.shape[1]

    w = np.array([lambdas[:P]**-1])

    # proiezione sugli assi principali ma prendendo solo le prime P coord per risparmiare tempo:
    x1_prime = x1.dot(E[:P].T)
    x2_prime = x2.dot(E[:P].T)

    if x1_prime.shape[1] != x2_prime.shape[1]:
        print('the two vectors must have same length')
        # puo' anche essere che x1 sia un vettore e x2 un db, ma il n di coord di ogni soggetto deve essere uguale
    else:
        #quad_coord = (x1_prime-x2_prime) * w * (x1_prime-x2_prime)         # looks like it takes about 50% more time
        quad_coord = w * (x1_prime-x2_prime)**2
        dist_quad = np.sum( quad_coord , axis=1 )
        return np.sqrt(dist_quad)


def relative_log_dist(d1, d2):
    if np.any(d1)==0 or np.any(d2)==0:
        print('error: both distances must be strictly positive')
    else:
        return np.log10(d1/d2)
