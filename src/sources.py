from numba import jit
import numpy as np
from constants import *

@jit
def plxsource(t, x, prl, pim, dt, Ein):
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
#       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per

        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for m in range(0,len(prl[0])-1):
                for q in range(0, len(prl[0][0]) - 1):
                

                        aaa = np.sin(2*np.pi*q/99)
                #            aaa = sin(2*pi*m/49)
                #            ptrans[m] = aaa
                        prl[x,m,q] = prl[x,m,q] + aaa*prl_add

@jit
def lxysource3d(t, x, y, prl, pim, dt, Ein):
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for q in range(0, len(prl[0][0]) - 1):
            

            aaa = np.sin(2*np.pi*q/99)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[x,y,q] = prl[x,y,q] + aaa*prl_add

@jit
def lxzsource3d(t, x, z, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for m in range(0, len(prl[0]) - 1):
            

            aaa = np.sin(2*np.pi*m/99)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[x,m,z] = prl[x,m,z] + aaa*prl_add

@jit
def lyzsource3d(t, y, z, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0, len(prl) - 1):
            

            aaa = np.sin(2*np.pi*i/99)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[i,y,z] = prl[i,y,z] + aaa*prl_add

@jit
def plysource(t, y, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0,len(prl) -1):
            for q in range(0, len(prl[0][0]) - 1):
                
    
                aaa = np.sin(2*np.pi*q/99)
        #            aaa = sin(2*pi*m/49)
        #            ptrans[m] = aaa
                prl[i,y,q] = prl[x,y,q] + aaa*prl_add

@jit
def plzsource(t, z, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0,len(prl) -1):
            for m in range(0, len(prl[0]) - 1):
                
    
                aaa = np.sin(2*np.pi*m/99)
        #            aaa = sin(2*pi*m/49)
        #            ptrans[m] = aaa
                prl[i,m,z] = prl[i,m,z] + aaa*prl_add

@jit
def psource3d(t, x, y, z, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
                
    
#            aaa = sin(2*pi*m/49)
#            ptrans[m] = aaa
        prl[x,y,z] = prl[x,y,z] + .05*prl_add
                
@jit
def psource2d(t, x, y, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
                

#            aaa = sin(2*pi*m/49)
#            ptrans[m] = aaa
        prl[x,y] = prl[x,y] + .05*prl_add

@jit
def psource1d(t, x, prl, pim, dt, Ein):
        T_per =h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

#            aaa = sin(2*pi*m/49)
#            ptrans[m] = aaa
        prl[x] = prl[x] + .05*prl_add

@jit
def lxsource2d(t, x, prl, pim, dt, Ein):
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for m in range(0, len(prl[0]) - 1):
            

            aaa = np.sin(2*np.pi*m/99)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[x,m] = prl[x,m] + aaa*prl_add

@jit
def lysource2d(t, y, prl, pim, dt, Ein):
        T_per = h_nobar_eV/(Ein*dt)
        sig = .65*T_per
    #       T_per =round( h_nobar/(Ein*dt),2)
        TC = 2*sig
        omg_in = 2*np.pi/T_per
        
        prl_add = 0.01*np.exp(-1.*((t-TC)/sig)**2)*np.cos(omg_in*(t-TC))

        #        ptrans = np.zeros(MM)
        for i in range(0, len(prl) - 1):
            

            aaa = np.sin(2*np.pi*i/99)
    #            aaa = sin(2*pi*m/49)
    #            ptrans[m] = aaa
            prl[i,y] = prl[i,y] + aaa*prl_add