###############################################################################
#: 
#: Helper functions used when building a simulation.
#: 
#: :filename: quantum.py
#: :author: Brandon Vetter <brandon.vetter@bvetter.org>
#: 
#: :summary: This module contains the helper functions for building simulations.
#: 
#: Property of University of Idaho
###############################################################################

from numba import jit
import numpy as np
from constants import *
import os
import cmath


def normalize_dft(dft):
    return np.abs(np.conj(dft)*dft)

def scale(V_drop, E_dft):
    return np.sqrt((E_dft + V_drop)/E_dft)

def transmission(inpt, output, scale=None, max_output=.01):
    if scale is None:
        scale = np.ones(len(inpt))
    
    max_value = max(inpt)
    max_value_per = max_value*max_output

    output = [o if o>max_value_per else 0 for o in output]
    return np.abs(output/inpt)*scale


def time_to_step(dt, time):
    return round(time/dt)

def step_to_time(dt, step):
    return dt*step

def dist_to_step(del_x, dist):
    return round(dist/del_x)

def step_to_dist(del_x, step):
    return del_x*step

def delta(x, loc, height):
    y = np.zeros(len(x))
    y[loc] = height
    return y

def calulate_mod(start, end, particle):
    psi = particle.prl + 1j*particle.pim
    ind = start
    mod = 0
    while ind < end:
        mod += psi[ind].conj*psi[ind]
        ind += 1

    return mod

def find_harmonic_diff(Ein, del_x, start=0, end=0, size=0, spatial=False):
    if spatial:
        start = round(start/del_x)
        end = round(end/del_x)
        size = round(size/del_x)
    if end == 0:
        mid = round(size/2)
        end = size
    if size == 0:
        size = end - start
        mid = round(size/2)
    slope = .5*m0*(Ein/hbar_J)**2*(del_x**2)
    drop = slope*(mid)**2

    return drop

def find_harmonic_Ein(drop, del_x, start=0, end=0, size=0, spatial=False):
    if spatial:
        start = round(start/del_x)
        end = round(end/del_x)
        size = round(size/del_x)
    if end == 0:
        mid = round(size/2)
        end = mid
    if size == 0:
        size = end - start
        mid = round(size/2)

    slope = (drop*eV2J)/(mid)**2
    Ein = np.sqrt(slope/(.5*m0*del_x**2))*hbar_J

    return Ein*J2eV

def H_matrix(NN, V, del_x):
    chi0 = 0.5*(hbar_J/m0)*(hbar_J/del_x**2)

    H = np.zeros( (NN,NN) )
    H[0,0] = 2*chi0 + V[0]
    H[0,1] = -1*chi0
    for n in range(1,NN-1):
        H[n,n-1] = -1*chi0
        H[n,n] = 2*chi0 + V[n]
        H[n,n+1] -1*chi0

    H[NN-1,NN-2]   = -1*chi0
    H[NN-1,NN-1] = 2*chi0 + V[NN-1]

    return H

def update_measurables(dementions, prl, pim, dx, V, iptot=None):
    """
    Calculate and update measurable physical quantities for the particle.

    :param dt: time step (not used directly here but kept for API compatibility)
    :param dx: spatial discretization step used for kinetic energy calculation
    :param V: potential array used to compute potential energy
    :returns: tuple (psi, ptot, ke, pe, E)
        - psi: complex wavefunction array
        - ptot: total probability
        - ke: kinetic energy
        - pe: potential energy
        - E: total energy
    """

    if iptot == None:
        ptot = np.sum(prl**2 + pim**2)
    else:
        ptot = iptot
    prl = prl/np.sqrt(ptot)
    pim = pim/np.sqrt(ptot)

    #ptot = np.sum(prl**2 + pim**2)
    psi = (prl + pim*1j)
    sim_size = prl.shape
    ke = 0    
    if dementions == 1:    
        for n in range(sim_size[0] -1):
            ke += (psi[n-1] - 2*psi[n] + psi[n+1])*np.conj(psi[n])

    if dementions == 2:    
        for i in range(sim_size[0] -1):
            for j in range(sim_size[1] - 1):
                ke += (psi[i-1][j] - 4*psi[i][j] + psi[i+1][j]
                                            +psi[i][j+1] + psi[i][j-1])*np.conjugate(psi[i,j])
    
    if dementions == 3:    
        for i in range(sim_size[0] -1):
            for j in range(sim_size[1] - 1):
                for q in range(sim_size[2] - 1):
                    ke += (psi[i-1][j][q] - 6*psi[i][j][q] + psi[i+1][j][q]
                                                + psi[i][j+1 ][q] + psi[i][j-1][q]
                                                + psi[i][j][q+1] + psi[i][j][q-1])*np.conj(psi[i,j,q])
                

                
    ke = (-J2eV*((hbar_J/dx)**2/(2*meff))*ke.real)
    pe = np.sum((prl**2 + pim**2)*V)*J2eV

    E = pe + ke

    return psi, ptot, ke, pe, E

def create_forier(samples):
    d_e = 1/samples
    E = np.arange(0, 1+d_e, d_e)
    forier = np.zeros(len(E), dtype=np.complex128)
    return E, forier
    
@jit(nopython=True)
def DFT(prl, pim, time, dt, E, forier):
    dimentions = len(np.shape(prl))
    # if dimentions == 1:
    #     for i in range(len(E)):
    #         forier[i] = (forier[i]+cmath.exp(-1j*(2*np.pi*E[i]/h_nobar_eV)*(time*dt)))*(
    #             prl[point] - 1j*pim[point])
    if dimentions == 2:
        
        for i in range(len(E)):
            for j in range(len(E[0])):
                forier[i][j] = (forier[i][j]+cmath.exp(-1j*(2*np.pi*E[i][j]/h_nobar_eV)*(time*dt)))*(
                    prl - 1j*pim)
                
    return forier

@jit
def descrete_forier_transform_point(dimentions, prl, pim, point, time, dt, E, forier):
    for i in range(len(E)):
        forier[i] = (forier[i]+cmath.exp(-1j*(2*np.pi*E[i]/h_nobar_eV)*c_time))*(prl[point] - 1j*pim[point])
    return forier


def descrete_forier_transform_square(dimentions, prl, pim, x, y, time, dt, E, forier):
    for i in range(len(E)):
        forier[i] = (forier[i]+cmath.exp(-1j*(2*np.pi*E[i]/h_nobar_eV)*c_time))*(np.sum(prl[x[0]:x[1]][y[0]:y[1]]) - 1j*np.sum(prl[x[0]:x[1]][y[0]:y[1]]))
    return forier



def find_ptot_range(prl, pim, x=None, y=None, z=None):
    ptot = np.sum(prl**2 + pim**2)
    sim_size = prl.shape
    ppt = 0
    if x is None and len(prl.shape) >= 1:
        x = [0,sim_size[0]]
    if y is None and len(prl.shape) >= 2:
        y = [0,sim_size[1]]
    if z is None and len(prl.shape) >= 3:
        z = [0,sim_size[2]]

    if len(sim_size) == 3:    
        ppt = np.sum(prl[x[0]:x[1]][y[0]:y[1]][z[0]:z[1]])
    if len(sim_size) == 2:    
        ppt = np.sum(prl[x[0]:x[1]][y[0]:y[1]])
    if len(sim_size) == 1:    
        ppt = np.sum(prl[x[0]:x[1]])
    
    return ptot/ppt
    
        
        
def cylindrical_to_cartesian(p, theta, z=None):
    x = p*np.cos(theta)
    y = p*np.sin(theta)
    if z is None:
        return x, y
    return x, y, z

def cartesian_to_cylindrical(x, y, z=None):
    p = x**2 + y**2
    theta = np.angle(y/x)
    if z is None:
        return x, y
    return x, y, z

def normalize(prl, pim):
    ptot = np.sum(prl**2 + pim**2)
    pnorm = np.sqrt(ptot)
    prl = np.divide(prl,pnorm)
    pim = np.divide(pim,pnorm)
    return prl, pim

def extract_vector(Vector, number):
    """
    Extracts a row vector
    """
    return Vector[:,number]


def pml(NN, cells = 50):
    nabc = cells
    xabc = np.ones(NN)
    show_pml = np.zeros(NN)
    for n in range(nabc,NN-nabc):
        show_pml[n] = 1

    for n in range(nabc + 1):
        xxn = (nabc - n) / nabc
    #    xpml[n] = 1 - .25 * xxn ** 3
        xabc[n] = 1 - .5 * xxn ** 3
        xabc[-n] = xabc[n]    
    return xabc

# fdtd simulations
@jit(parallel=True)
def fdtd(prl, pim, n_step, ra, rd, V=None, pml=None, pml_args = []):
    """
    This is the FDTD method. This function is not used.  It can do 
    1d simulations. 
    """
    NN = len(prl)
    if V == None:
        V = np.zeros(NN)
    if pml == None:
        abc = np.ones(NN)
    else:
        abc = pml(*pml_args)
    print("FDTD: nstep = ",n_step)
    for _ in range(n_step):
        prl[1] = abc[1]*prl[1] - ra*(pim[0] - 2*pim[1] + pim[2]) + rd*V[1]*pim[1]
        for n in range(2, NN-2):
            prl[n] = abc[n]*prl[n] - ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*V[n]*pim[n]
            pim[n-1] = abc[n-1]*pim[n-1] + ra*(prl[n-2] - 2*prl[n-1] + prl[n]) - rd*V[n-1]*prl[n-1]
            
        pim[NN-2] = abc[NN-2]*pim[NN-2] + ra*(prl[NN-3] - 2*prl[NN-2] + prl[NN-1]) - rd*V[NN-2]*prl[NN-2]

    return prl, pim

def _kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():
    """
    Stole this from a stack overflow on killing the cache.  
    https://stackoverflow.com/questions/44131691/how-to-clear-cache-or-force-recompilation-in-numba
    """
    root_folder = os.path.realpath(__file__ + "/../../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)
