"""
This file contains 2d methods for E fields.

:Author: Brandon Vetter
:Date: 2/15/26

Property of the University of Idaho
"""

from numba import jit
import matplotlib.pyplot as plt
import numpy as np
from constants import *

def sten(V, E=1, force=False):
    """
    fills the V field with a set value, can force it to a specific value
    
    :param V: V field
    :param E=1: What to add or set value to
    :param force=False: If to force to specific value or add it
    """

    if force == True:
        V = E*np.ones(V.shape)*eV2J
    else:
        V += E*np.ones(V.shape)*eV2J

def draw_cylender(V, center_x, center_y, radius, E, force=False):
    """
    draws a cylinder in the V field

    :param V: V field
    :param center_x: x center of the cylinder
    :param center_y: y center of the cylinder
    :param radius: radius of the cylinder
    :param E: value to set or add
    :param force=False: If to force to specific value or add it
    """
    for i in range(len(V)-1):
        for j in range(len(V[0])-1):
            rad = (i - center_x)**2 + (j - center_y)**2
            if rad <= radius**2:
                if force:
                    V[i][j] = E*eV2J
                else:
                    V[i][j] += E*eV2J

def draw_hollow_cylender(V, center_x, center_y, radius, value, force=False):
    """
    draws a cylinder in the V field

    :param V: V field
    :param center_x: x center of the cylinder
    :param center_y: y center of the cylinder
    :param radius: radius of the cylinder
    :param E: value to set or add
    :param force=False: If to force to specific value or add it
    """
    for i in range(len(V)-1):
        for j in range(len(V[0])-1):
            rad = (i - center_x)**2 + (j - center_y)**2
            if rad == radius**2:
                if force:
                    V[i][j] = value
                else:
                    V[i][j] += value

def draw_circular_exp(V, center_x, center_y, radius, alpha, force=False):
    """
    draws a circular exponential in the V field

    :param V: V field
    :param center_x: x center of the circle
    :param center_y: y center of the circle
    :param radius: radius of the circle
    :param alpha: exponential decay factor
    :param force=False: If to force to specific value or add it
    """

    for i in range(len(V)-1):
        for j in range(len(V[0])-1):
            rad = (i - center_x)**2 + (j - center_y)**2
            if rad <= radius**2:
                if force:
                    V[i][j] = (-np.exp(-alpha*np.abs(i - center_x)) - np.exp(-alpha*np.abs(j - center_y)))*eV2J
                else:
                    V[i][j] += (-np.exp(-alpha*np.abs(i - center_x)) - np.exp(-alpha*np.abs(j - center_y)))*eV2J

def draw_circular_coulmbc(V, center_x, center_y, radius, qc, del_x, di=perm, force=False):
    """
    draws a circular Coulomb potential in the V field

    :param V: V field
    :param center_x: x center of the circle
    :param center_y: y center of the circle
    :param radius: radius of the circle
    :param qc: charge value
    :param force=False: If to force to specific value or add it
    """

    for i in range(len(V)):
        for j in range(len(V[0])):
            rad = (i - center_x)**2 + (j - center_y)**2
            if rad <= radius**2:
                try:
                    if force:
                        V[i][j] = (((qc)/(4*np.pi*di*del_x*np.sqrt((i-center_x)**2+(j-center_y)**2))))*eV**2
                    else:
                        V[i][j] += (((qc)/(4*np.pi*di*del_x*np.sqrt((i-center_x)**2+(j-center_y)**2))))*eV**2
                except ZeroDivisionError:
                    continue

def draw_Efield_circle(x, y, V):
    """
    draws E field circle
    :param x: x coordinate
    :param y: y coordinate
    :param V: V field
    """
    equation = np.exp(-V)