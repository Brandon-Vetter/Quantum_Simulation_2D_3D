from numba import jit
import numpy as np
from constants import *


def abc1d(sim_space, start_loc, dec=.5):


    ret_abc = np.ones(len(sim_space))
    for n in range(round(len(sim_space)/2)):
        if n < start_loc:
            ret_abc[n] = 1 - dec*((start_loc - start_loc)/start_loc)**3
        if (len(sim_space)-1) - n > len(sim_space) - start_loc:
            ret_abc[(len(sim_space)-1) - n] = 1 - dec*((start_loc - start_loc)/start_loc)**3
    return ret_abc

def abcx2d(sim_space, start_loc, dec=.5):
    pml = np.ones(sim_space.shape)
    
    for n in range(start_loc + 1):
        xxn = (start_loc - n) / start_loc
        pml[n][:] = 1 - .5 * xxn ** 3
        pml[-n][:] = pml[n][:]    
    
    return pml

def abcy2d(sim_space, start_loc, dec=.5):
    pml = np.ones(sim_space.shape).T
    
    for n in range(start_loc + 1):
        xxn = (start_loc - n) / start_loc
        pml[:][n] = 1 - .5 * xxn ** 3
        pml[:][-n] = pml[:][n]
    
    return pml.T

def abcx3d(sim_space, start_loc, dec=.5):
    pml = np.ones(sim_space.shape)
    
    for n in range(start_loc + 1):
        xxn = (start_loc - n) / start_loc
        pml[n][:][:] = 1 - .5 * xxn ** 3
        pml[-n][:][:] = pml[n][:][:]    
    
    return pml

def abcy3d(sim_space, start_loc, dec=.5):
    pml = np.ones(sim_space.shape)
    
    for n in range(start_loc + 1):
        xxn = (start_loc - n) / start_loc
        pml[:][n][:] = 1 - .5 * xxn ** 3
        pml[:][-n][:] = pml[:][n][:]
    
    return pml

def abcz3d(sim_space, start_loc, dec=.5):
    pml = np.ones(sim_space.shape)
    
    for n in range(start_loc + 1):
        xxn = (start_loc - n) / start_loc
        pml[:][:][n] = 1 - .5 * xxn ** 3
        pml[:][:][-n] = pml[:][:][n]
    
    return pml