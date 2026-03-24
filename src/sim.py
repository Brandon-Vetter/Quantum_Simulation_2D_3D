##############################################################################
#: This file contains the fdtd methods.  Contains 1D, 2D, and 3D methods.
#:
#: :Author: Brandon Vetter
#: :Date: 2/15/26
#:
#: Property of the University of Idaho
##############################################################################

from numba import jit
import sys
import numpy as np
import matplotlib.pyplot as plt
import sources
import aabc
import vfields.d2 as vf
from matplotlib import cm
import matplotlib.animation as animation
from functools import partial
from constants import *

@jit
def misc(t, prl, pim, V, ra, rd, part_id):
    """
    This is a userdefined function.  It should be overwritten.  Input any code
        to run in the FDTD here (like forour transform etc)
    :returns: none
    
    Todo: Implement this as a yaml file
    """
    pass

@jit
def source(t, prl, pim, V, ra, rd, part_id):
    """
    User Defined Source.  Should be overwritten.  Use Source library to build
        your own source.
    :returns: none
    
    Todo: implement this as a yaml file.
    """
    pass


def fdtdndmp(steps, dimentions, particles, V, ra, rd, del_x, t=0, abc=None, attract=1):
    """
    Is the wrapper for the fdtd that rans partilces with the hartree approximation
    
    :param steps: how many steps to run simulation for
    :param dimentions: how dimentions is the simulation
    :param partilces: dictonary of all particles in sim.  Key is particle ID, 
        value is a tuple of (prl, pim)
    :param V: V field to add to the simulation
    :param ra: ra value for the fdtd method
    :param rd: rd value for the fdtd method
    :param t: time value for rdtd method, default = 0
    :param abc: is the pml. Default     
    
    :returns: Returns the coloumb effect for every particle
    """
    kernal = _hartree_kernal(V.shape, del_x)
    Fkernal = np.fft.fft(kernal)
    time = t
    hf_part = {}
    
    
    # setup the columb E field to 0 for each particle
    for part_id in particles.keys():
        hf_part[part_id] = np.zeros(V.shape, dtype=np.float64)
    for step in range(steps):
        for part_id, part in particles.items():
            # iterate though each of the particles
            hf = np.zeros(V.shape, dtype=np.float64) # set the columb effect for this particle to 0
            for part_hf_id, hf_value in hf_part.items():
                # iterate though each particle's columb effect
                if part_hf_id == part_id:
                    # if the current particle is the current columb effect, skip
                    continue
                # add the current particle's colomb effect to total columb
                # add in passed E field to columb verable
                hf += hf_value
                hf += V


            # run FDTD for where hf is the E field + partilce's colomb effect
            if dimentions == 1:
                _fdtd1dl(1, part[0], part[1], hf, abc, ra, rd, t=time, part_id = part_id)
            elif dimentions == 2:
                _fdtd2dl(1, part[0], part[1], hf, abc, ra, rd, t=time, part_id = part_id)
            elif dimentions == 3:
                _fdtd3dl(1, part[0], part[1], hf, abc, ra, rd, t=time, part_id = part_id)
            else:
                _fdtd1dl(1, part[0], part[1], hf, abc, ra, rd, t=time, part_id = part_id)
        
        # calulate the columb effect for each particle
        for part_id, part in particles.items():
            hp = part[0]**2 + part[1]**2
            HP = np.fft.fft(hp)
            HPT = Fkernal[0:int(len(Fkernal)/2)] * HP
            hf = attract*np.fft.ifft(HPT)
            hf_part[part_id] = hf.real
            
        time += 1
        
    # return the columb effect for every particle
    for part_hf_id, hf_value in hf_part.items():
        try:
            hf += hf_value
        except UnboundLocalError:
            hf = None
            break
    return time, hf
        
    
def fdtdnd(steps, dimentions, prl, pim, V, ra, rd, t=0, abc=None, hf = None, part_id = 0):
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
    if abc is None:
        abc = np.ones(prl.shape)

    if dimentions == 1:
        return _fdtd1dl(steps, prl, pim, V, abc, ra, rd, t=t, part_id = part_id)
    elif dimentions == 2:
        return _fdtd2dl(steps, prl, pim, V, abc, ra, rd, t=t, part_id = part_id)
    elif dimentions == 3:
        return _fdtd3dl(steps, prl, pim, V, abc, ra, rd, t=t, part_id = part_id)
    else:
        return _fdtd1dl(steps, prl, pim, V, abc, ra, rd, t=t, part_id = part_id)

@jit
def _fdtd1dl(steps, prl, pim, V, abc, ra, rd, t=0, part_id = 0):
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
        source(t, prl, pim, V, ra, rd, part_id)
        misc(t, prl, pim, V, ra, rd, part_id)
        

    return t

@jit
def _fdtd2dl(steps, prl, pim, V, abc, ra, rd, t=0, part_id = 0):
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
        source(t, prl, pim, V, ra, rd, part_id)
        misc(t, prl, pim, V, ra, rd, part_id)
    
    return t
        
@jit    
def _fdtd3dl(steps, prl, pim, V, abc, ra, rd, t=0, part_id = 0):
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
        source(t, prl, pim, V, ra, rd, part_id)
        misc(t, prl, pim, V, ra, rd, part_id)
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
        prl[n] = abc[n]*prl[n] - ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*(V[n])*pim[n]
    
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
    sigma = .2
    lambd = .01*eV2J
    X_size = 4000
    Y_size = 400
    Z_size = 100
    del_x = 0.1e-9
    dt = 1e-17
    ra = (0.5*hbar_J/meff)*(dt/del_x**2)
    rd = dt/hbar_J
    
    @jit 
    def source(t, prl, pim, V, ra, rd, part_id):
        if part_id == 0:
            sources.lxsource2d(t, 600, prl, pim, dt, .02)
        else:
            sources.lxsource2d(t, 3200, prl, pim, dt, .02)
    

    x = np.linspace(0, X_size*del_x, X_size)
    y = np.linspace(0, Y_size*del_x, Y_size)
    X, Y = np.meshgrid(x, y)
    
    particles = {}
    prl1 = np.zeros((X_size, Y_size))
    pim1 = np.zeros((X_size, Y_size))
    particles[0] = (prl1,pim1)
    
    prl2 = np.zeros((X_size, Y_size))
    pim2 = np.zeros((X_size, Y_size))
    particles[1] = (prl2,pim2)
    

    
    V = np.zeros((X_size, Y_size))
    user_ip = ""
    hf = np.zeros(len(V))
    time = 0
    while True:

        user_ip = int(input("time step: "))
        if user_ip == 0:
            break
        time, hf = fdtdndmp(user_ip, 2, particles, V, ra, rd, abc=pml, t=time, attract=-1)
        fig, ax = plt.subplots(projection="3d")
    
        surf = ax.plot_surface(X*1E9, Y*1E9, prl)
        plt.show()

    sys.exit()
        
    
    fig, ax = plt.subplots(2,1)
    ax[0].grid()
    ax[1].grid()
    prlp1, = ax[0].plot(x*1E9, prl1, "b", label="P1 real")
    pimp1, = ax[0].plot(x*1E9, pim1, "--r", label="P1 imaginary")
    prlp2, = ax[0].plot(x*1E9, prl2, "orange", label="P2 real")
    pimp2, = ax[0].plot(x*1E9, pim2, "--c", label="P2 imaginary")
    time_text = fig.text(.1,.01,f"{time*dt*1E12 : .4f}ps")
    hfp, = ax[1].plot(x*1E9, hf*J2eV, label="coloumb")
    ax[0].legend(loc=(1.01, .2))
    ax[1].set_title("columb energy")
    ax[0].set_title("Particles")
    ax[1].legend(loc=(1.01, .5))
    plt.tight_layout()
    
    ax[1].set_ylim(-.1,1)
    ax[0].set_ylim(-.01,.01)
    def update(frame):
        global time
        time, hf = fdtdndmp(100, 1, particles, V, ra, rd, abc=pml, t=time, attract=-1)
        
        #print(f"{time*dt*1E12}ps")
        prlp1.set_ydata(prl1)
        pimp1.set_ydata(pim1)
        prlp2.set_ydata(prl2)
        pimp2.set_ydata(pim2)
        hfp.set_ydata(hf*J2eV)
        time_text.set_text(f"{time*dt*1E12 : .4f}ps")
        return time
        

    sim_time = 3.5E-12
    fdtd_update = 100
    frame_step = dt*fdtd_update
    frames_to_set_time = int(sim_time/frame_step)
    ani = animation.FuncAnimation(fig, partial(update), frames=frames_to_set_time, interval=50)
    plt.show()
    ani.save("coulumb_animation_2.mp4")
        