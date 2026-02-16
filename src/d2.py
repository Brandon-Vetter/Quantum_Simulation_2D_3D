from numba import jit
import matplotlib.pyplot as plt
import numpy as np
from constants import *

def sten(V, E=1, force=False):

    if force == True:
        V = E*np.ones(V.shape)*eV2J
    else:
        V += E*np.ones(V.shape)*eV2J

def draw_cylender(V, center_x, center_y, radius, E, force=False):
    for i in range(len(V)-1):
        for j in range(len(V[0])-1):
            rad = (i - center_x)**2 + (j - center_y)**2
            if rad <= radius**2:
                if force:
                    V[i][j] = E*eV2J
                else:
                    V[i][j] += E*eV2J

def draw_circular_exp(V, center_x, center_y, radius, alpha, force=False):
    for i in range(len(V)-1):
        for j in range(len(V[0])-1):
            rad = (i - center_x)**2 + (j - center_y)**2
            if rad <= radius**2:
                if force:
                    V[i][j] = (-np.exp(-alpha*np.abs(i - center_x)) - np.exp(-alpha*np.abs(j - center_y)))*eV2J
                else:
                    V[i][j] += (-np.exp(-alpha*np.abs(i - center_x)) - np.exp(-alpha*np.abs(j - center_y)))*eV2J

def draw_Efield_circle(x, y, V):
    equation = np.exp(-V)