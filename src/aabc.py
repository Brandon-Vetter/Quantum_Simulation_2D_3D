from numba import jit
import numpy as np
from constants import *


def abc1d(sim_space, start_loc, dec=.5):
    """
    Create a pml for 1 demention

    :param sim_space: a sim space varable to get dimentions
    :param start_loc: where to start the pml
    :param dec: how much to attenuate the simulation

    :returns: a pml value
    """
    pml = np.ones(sim_space.shape)
    
    for n in range(start_loc + 1):
        xxn = (start_loc - n) / start_loc
        pml[n] = 1 - .5 * xxn ** 3
        pml[-n] = pml[n]    
    
    return pml

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