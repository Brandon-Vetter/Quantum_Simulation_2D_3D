##############################################################################
#: This file contains all the source methods.  Contains 1D, 2D, and 3D methods.
#:
#: :Author: Brandon Vetter
#: :Date: 2/15/26
#:
#: Property of the University of Idaho
##############################################################################

from numba import jit
import numpy as np
from constants import *

@jit
def plxsource(t, x, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        plane source on a specific x plane

        :param t: time
        :param x: Point to set plane
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
#       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per

        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for m in range(0,len(prl[0])-1):
                for q in range(0, len(prl[0][0]) - 1):
                

                        if T_len == 0:
                            aaa = 1
                        else:
                            w = np.pi/(2*T_len)
                            aaa = np.sin(q*w + phase_shift)
                #            aaa = sin(2*pi*m/49)
                #            ptrans[m] = aaa
                        prl[x,m,q] = prl[x,m,q] + aaa*prl_add

@jit
def lxysource3d(t, x, y, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        line source on xy plane in 3d

        :param t: time
        :param x: x coordinate
        :param y: y coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for q in range(0, len(prl[0][0]) - 1):
            

            if T_len == 0:
                aaa = 1
            else:
                w = np.pi/(2*T_len)
                aaa = np.sin(q*w + phase_shift)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[x,y,q] = prl[x,y,q] + aaa*prl_add

@jit
def lxzsource3d(t, x, z, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        line source on xz plane in 3d

        :param t: time
        :param x: x coordinate
        :param z: z coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for m in range(0, len(prl[0]) - 1):
            

            if T_len == 0:
                aaa = 1
            else:
                w = np.pi/(2*T_len)
                aaa = np.sin(m*w + phase_shift)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[x,m,z] = prl[x,m,z] + aaa*prl_add

@jit
def lyzsource3d(t, y, z, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        line source on yz plane in 3d

        :param t: time
        :param y: y coordinate
        :param z: z coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0, len(prl) - 1):
            

            if T_len == 0:
                aaa = 1
            else:
                w = np.pi/(2*T_len)
                aaa = np.sin(i*w + phase_shift)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[i,y,z] = prl[i,y,z] + aaa*prl_add

@jit
def plysource(t, y, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        plane source on a specific y plane

        :param t: time
        :param y: Point to set plane
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0,len(prl) -1):
            for q in range(0, len(prl[0][0]) - 1):
                
    
                if T_len == 0:
                    aaa = 1
                else:
                    w = np.pi/(2*T_len)
                    aaa = np.sin(q*w + phase_shift)
        #            aaa = sin(2*pi*m/49)
        #            ptrans[m] = aaa
                prl[i,y,q] = prl[i,y,q] + aaa*prl_add

@jit
def plzsource(t, z, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        plane source on a specific z plane

        :param t: time
        :param z: Point to set plane
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0,len(prl) -1):
            for m in range(0, len(prl[0]) - 1):
                
    
                if T_len == 0:
                    aaa = 1
                else:
                    w = np.pi/(2*T_len)
                    aaa = np.sin(m*w + phase_shift)
        #            aaa = sin(2*pi*m/49)
        #            ptrans[m] = aaa
                prl[i,m,z] = prl[i,m,z] + aaa*prl_add

@jit
def psource3d(t, x, y, z, prl, pim, dt, Ein):
        """
        point source in 3d

        :param t: time
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        aaa = 1
                
    
#            aaa = sin(2*pi*m/49)
#            ptrans[m] = aaa
        prl[x,y,z] = prl[x,y,z] + aaa*prl_add
                
@jit
def psource2d(t, x, y, prl, pim, dt, Ein):
        """
        point source in 2d

        :param t: time
        :param x: x coordinate
        :param y: y coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        aaa = 1
                

#            aaa = sin(2*pi*m/49)
#            ptrans[m] = aaa
        prl[x,y] = prl[x,y] + aaa*prl_add

@jit
def psource1d(t, x, prl, pim, dt, Ein):
        """
        point source in 1d

        :param t: time
        :param x: x coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        aaa = 1
#            aaa = sin(2*pi*m/49)
#            ptrans[m] = aaa
        prl[x] = prl[x] + aaa*prl_add

@jit
def lxsource2d(t, x, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        line source on x in 2d

        :param t: time
        :param x: x coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for m in range(0, len(prl[0]) - 1):
            

            if T_len == 0:
                aaa = 1
            else:
                w = np.pi/(2*T_len)
                aaa = np.sin(m*w + phase_shift)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[x,m] = prl[x,m] + aaa*prl_add

@jit
def lysource2d(t, y, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        line source on y in 2d

        :param t: time
        :param y: y coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)

        for i in range(0, len(prl) - 1):
            

            if T_len == 0:
                aaa = 1
            else:
                w = np.pi/(2*T_len)
                aaa = np.sin(i*w + phase_shift)
            prl[i,y] = prl[i,y] + aaa*prl_add

@jit
def lysource2d_gauss(t, y, prl, pim, dt, Ein, T_len = 0, phase_shift=0):
        """
        line source on y in 2d

        :param t: time
        :param y: y coordinate
        :param prl: Real part of Schrödinger
        :param pim: Imaginary part of Schrödinger
        :param dt: Change in time
        :param Ein: Input energy (larger value will take longer to output)
        """
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)

        #        ptrans = np.zeros(MM)

        for i in range(0, len(prl) - 1):
            

            if T_len == 0:
                aaa = 1
            else:
                w = np.pi/(2*T_len)
                aaa = np.sin(i*w + phase_shift)
            prl[i,y] = prl[i,y] + aaa*prl_add