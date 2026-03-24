"""
This file contains the fdtd methods.  Contains 1D, 2D, and 3D methods.

:Author: Brandon Vetter
:Date: 2/15/26
:Licence: Apatche
"""
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import sources
import aabc
import vfields.d2 as vf
from matplotlib import cm
from constants import *

@jit
def misc(t, prl, pim, V, ra, rd):
    """
    This is a userdefined function.  It should be overwritten.  Input any code
    to run in the FDTD here (like forour transform etc)
    :return: none
    
    Todo: Implement this as a yaml file
    """
    pass

@jit
def source(t, prl, pim, V, ra, rd):
    """
    User Defined Source.  Should be overwritten.  Use Source library to build
    your own source.
    :returns: none
    
    Todo: implement this as a yaml file.
    """
    pass

def fdtdnd(steps, dimentions, prl, pim, V, ra, rd, t=0, abc=None):
    if abc is None:
        abc = np.ones(prl.shape)

    if dimentions == 1:
        return _fdtd1dl(steps, prl, pim, V, abc, ra, rd, t=t)
    elif dimentions == 2:
        return _fdtd2dl(steps, prl, pim, V, abc, ra, rd, t=t)
    elif dimentions == 3:
        return _fdtd3dl(steps, prl, pim, V, abc, ra, rd, t=t)
    else:
        return _fdtd1dl(steps, prl, pim, V, abc, ra, rd, t=t)

@jit
def _fdtd1dl(steps, prl, pim, V, abc, ra, rd, t=0):
    """
    Base function for fdtd methods.  Manages time, and figures out what fdtd method to run
    :param steps: how many steps to run simulation for
    :param dimentions: how dimentions is the simulation
    :param prl: real part of the shrodinger equation.  Is pass by referance
    :param pim: imaginary part of the shrodinger equation.  Is pass by referance
    :param V: V field to add to the simulation
    :param abc: is the pml
    :param ra: ra value for the fdtd method
    :param rd: rd value for the fdtd method
    :param t: time value for rdtd method, default = 0
    
    :returns: Returns the elapsed time of the simulation
    """
    for step in range(steps):
        _fdtd1d(prl, pim, V, abc, ra, rd)

        t += 1
        source(t, prl, pim, V, ra, rd)
        misc(t, prl, pim, V, ra, rd)
        

    return t

@jit
def _fdtd2dl(steps, prl, pim, V, abc, ra, rd, t=0):
    """
    Base function for fdtd methods.  Manages time, and figures out what fdtd method to run
    :param steps: how many steps to run simulation for
    :param dimentions: how dimentions is the simulation
    :param prl: real part of the shrodinger equation.  Is pass by referance
    :param pim: imaginary part of the shrodinger equation.  Is pass by referance
    :param V: V field to add to the simulation
    :param abc: is the pml
    :param ra: ra value for the fdtd method
    :param rd: rd value for the fdtd method
    :param t: time value for rdtd method, default = 0
    
    :returns: Returns the elapsed time of the simulation
    """
    for step in range(steps):
        _fdtd2d(prl, pim, V, abc, ra, rd)

        t += 1
        source(t, prl, pim, V, ra, rd)
        misc(t, prl, pim, V, ra, rd)
    
    return t
        
@jit    
def _fdtd3dl(steps, prl, pim, V, abc, ra, rd, t=0):
    """
    Base function for fdtd methods.  Manages time, and figures out what fdtd method to run
    :param steps: how many steps to run simulation for
    :param dimentions: how dimentions is the simulation
    :param prl: real part of the shrodinger equation.  Is pass by referance
    :param pim: imaginary part of the shrodinger equation.  Is pass by referance
    :param V: V field to add to the simulation
    :param abc: is the pml
    :param ra: ra value for the fdtd method
    :param rd: rd value for the fdtd method
    :param t: time value for rdtd method, default = 0
    
    :returns: Returns the elapsed time of the simulation
    """
    for step in range(steps):
        _fdtd3d(prl, pim, V, abc, ra, rd)

        t += 1
        source(t, prl, pim, V, ra, rd)
        misc(t, prl, pim, V, ra, rd)
    return t

@jit
def _fdtd3d(prl, pim, V, abc, ra, rd):
    """
    3d shordinger equation
    
    :param prl: Is the real part of the shrodinger equation.  Must be 3 dimentions
    :param pim: Is the imagainary part of the shrodinger equation.  Must be 3 dimentions
    :param abc: Is the pml around the simulation in any
    :param V: Is the E field of the shrodinger equation.  Must be 3 dimentions
    :param ra: Is the ra value
    :param rd: Is the rd value

    :returns: None
    """
    for i in range(len(prl)-1):
        for j in range(len(prl[i])-1):
            for q in range(len(prl[i][j])-1):
                prl[i][j][q] = abc[i][j][q]*prl[i][j][q] - ra*(pim[i-1][j][q] - 6*pim[i][j][q] + pim[i+1][j][q]
                                            + pim[i][j-1][q] + pim[i][j+1][q]
                                            + pim[i][j][q+1] + pim[i][j][q-1]) + rd*V[i][j][q]*pim[i][j][q]
    
    for i in range(len(pim)-1):
        for j in range(len(pim[i])-1):
            for q in range(len(pim[i][j])-1):
                pim[i][j][q] = abc[i][j][q]*pim[i][j][q] + ra*(prl[i-1][j][q] - 6*prl[i][j][q] + prl[i+1][j][q]
                                            + prl[i][j+1 ][q] + prl[i][j-1][q]
                                            + prl[i][j][q+1] + prl[i][j][q-1]) - rd*V[i][j][q]*prl[i][j][q]

@jit
def _fdtd2d(prl, pim, V, abc, ra, rd):
    """
    2d shordinger equation
    
    :param prl: Is the real part of the shrodinger equation.  Must be 2 dimentions
    :param pim: Is the imagainary part of the shrodinger equation.  Must be 2 dimentions
    :param V: Is the E field for the shrodinger equation.  Must be 2 dimentions
    :param abc: Is the pml around the simulation in any
    :param ra: Is the ra value
    :param rd: Is the rd value

    :returns: None
    """
    for i in range(len(prl)-1):
        for j in range(len(prl[i])-1):
            prl[i][j] = abc[i][j]*prl[i][j] - ra*(pim[i-1][j] - 4*pim[i][j] + pim[i+1][j]
                                        + pim[i][j-1] + pim[i][j+1]) + rd*V[i][j]*pim[i][j]
    
    for i in range(len(pim)-1):
        for j in range(len(pim[i])-1):
            pim[i][j] = abc[i][j]*pim[i][j] + ra*(prl[i-1][j] - 4*prl[i][j] + prl[i+1][j]
                                        +prl[i][j+1] + prl[i][j-1]) - rd*V[i][j]*prl[i][j]
            
@jit
def _fdtd1d(prl, pim, V, abc, ra, rd):
    """
    1d shordinger equation
    
    :param prl: Is the real part of the shrodinger equation.  Must be 1 dimention
    :param pim: Is the imagainary part of the shrodinger equation.  Must be 1 dimention
    :param abc: Is the pml around the simulation in any
    :param V: Is the E field for the shrodinger equation.  Must be 1 dimention
    :param ra: Is the ra value
    :param rd: Is the rd value

    :returns: None
    """
    for n in range(len(prl)-1):
        prl[n] = abc[n]*prl[n] - ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*V[n]*pim[n]
    
    for n in range(len(pim)-1):
        pim[n] = abc[n]*pim[n] + ra*(prl[n-1] - 2*prl[n] + prl[n+1]) - rd*(V[n])*prl[n]

@jit
def _hartree_kernal1d(NN, del_x, di=perm):
    """
    Creates a kernal of all values for the hartree approximation to convolve
    with the particles.  For one dimention of particles
    
    :param NN: size of the simulation
    :param del_x: change in x
    :param abc: Is the pml around the simulation in any
    :param di: ebsolon value for material

    :returns: hartree_kernal
    """
    const = (eV)**2/(4*np.pi*di*del_x)
    hartree_kernal = np.zeros(2*NN, dtype=np.complex128)
    hartree_kernal[0] = const
    hartree_kernal[-1] = const
    for i in range(1, NN-1):
        hartree_kernal[i] = const/(0-i)
        hartree_kernal[-i] = const/(0-i)
    
    return hartree_kernal*const

def _hartree_kernal(dementions, del_x, di=perm):
    if len(dementions) == 1:
        return _hartree_kernal1d(dementions[0], del_x, di=perm)
    
    if len(dementions) == 2:
        harx = _hartree_kernal1d(dementions[0], del_x, di=perm)
        hary = _hartree_kernal1d(dementions[1], del_x, di=perm)
        HX, HY = np.meshgrid(harx, hary)
        return HX, HY

    if len(dementions) == 3:
        harx = _hartree_kernal1d(dementions[0], del_x, di=perm)
        hary = _hartree_kernal1d(dementions[1], del_x, di=perm)
        harz = _hartree_kernal1d(dementions[2], del_x, di=perm)
        HX, HY, HZ = np.meshgrid(harx, hary, harz)
        return HX, HY, HZ

@jit
def folk1d(prl1, pim1, prl2, pim2, del_x, p1_spin, p2_spin, di=si_di):
    kronecker = 1 if p1_spin == p2_spin else 0
    const = (eV)**2/(4*np.pi*di)
    psi1 = prl1 + pim1*1j
    psi2 = prl2 + pim2*1j
    hartree = np.zeros(len(prl1), dtype=np.complex128)
    for i in range(len(hartree)):
        for j in range(len(prl2)):
            if (i - j != 0):
                hartree[i] += (((psi2[j])**2)/(del_x*(i - j)))
                - ((np.conjugate(psi1[j])*psi2[j])/(del_x*(i - j)))*kronecker
    
    return hartree*const
        
if __name__ == "__main__":
    dt = 5e-17
    @jit 
    def source(t, prl, pim, V, ra, rd):
        sources.lysource2d(t, 60, prl, pim, dt, .3)
    sigma = .5
    lambd = .01*eV2J
    X_size = 300
    Y_size = 200
    Z_size = 50
    del_x = 0.2e-9

    ra = (0.5*hbar_J/meff)*(dt/del_x**2)
    rd = dt/hbar_J
    
    x = np.linspace(0, X_size*del_x, X_size)
    y = np.linspace(0, Y_size*del_x, Y_size)
   # z = np.linspace(0, Z_size*del_x, Z_size)
    
    X, Y = np.meshgrid(x, y)
    
    pml = aabc.abcy2d(X, 50)

    prl = np.zeros((Y_size,X_size))
    pim = np.zeros((Y_size,X_size))
    V = np.zeros((Y_size,X_size))
    vf.sten(V, 1)
    vf.draw_circular_exp(V, 100, 200, 60, .1)
    vf.draw_cylender(V, 100, 200, 40, 0, force=True)
    time = 0
    
    user_ip = ""
    while user_ip != "0":
        user_ip = input("time step: ")
        
        try:
            time = fdtdnd(int(user_ip), 2, prl, pim, V, ra, rd, t=time, abc=pml)
        except ValueError:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        surf = ax.plot_surface(X*1E9, Y*1E9, V*J2eV, rstride =10, cstride = 1, cmap=cm.plasma, linewidth = .4)
        #    ax.view_init(elev=30., azim = 30)
        ax.view_init(elev=30., azim = -75)
         #   plt.text(10,10,.2,"T = : {}".format(round(T,0)) )
        plt.ylabel('Y (nm)')
         #   ax.text2D(5,5,"T = 0")
        plt.xlabel('X (nm)')    
    
        plt.savefig("3dexample.png")
        plt.show()