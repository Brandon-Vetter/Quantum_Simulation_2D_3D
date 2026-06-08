import sys
sys.path.append("../src")

from numba import jit, njit
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import sources
from sim import *
import sim
import vfields.d2 as vf
import vfields.d1
import quantum
import aabc
from constants import *
import inspect
import sys

@jit 
def source(t, prl, pim, V, ra, rd, part_id):
    sources.lysource2d(t, 55, prl, pim, dt, .12, T_len = 100, phase_shift=np.pi)

@jit
def misc(t, prl, pim, V, ra, rd, part_id):
    if part_id != 1:
        prl *= metal
        pim *= metal
    


sim.source = source
sim.misc = misc

X_size = 200
Y_size = 200
Z_size = 100
samples = 1000
del_x = 0.2e-9


Metal_locx = int(X_size/2)
Metal_locy = int(Y_size/2)
dt = .5e-16
ra = (0.5*hbar_J/meff)*(dt/del_x**2)
rd = dt/hbar_J
sigma = .5
lambd = .01*eV2J


x = np.linspace(0, X_size*del_x, X_size)
y = np.linspace(0, Y_size*del_x, Y_size)

X, Y = np.meshgrid(y, x)

pim_1d = np.zeros(X_size)
prl_1d = np.zeros(X_size)



metal = np.ones((X_size, Y_size))
V = np.zeros((X_size, Y_size))
V_test = np.zeros((Y_size,X_size))

vf.draw_cylender(metal, Metal_locx, Metal_locy, 4.8, 0, force=True)
vf.draw_circular_coulmbc(V, Metal_locx, Metal_locy, 200, -1, del_x)
vf.draw_cylender(V, Metal_locx, Metal_locy, 4.8, 1000, force=True) # only for representation


well_slice = V[Metal_locy][:]


H = quantum.H_matrix(int(X_size/2), well_slice[:int(X_size/2)], del_x)
eigen_values, eigen_vectors = np.linalg.eigh(H)

eigentstate = quantum.extract_vector(eigen_vectors, 0)


vfields.d1.mirror(eigentstate, prl_1d)

prl, pim = quantum.normalize(pim_1d, prl_1d)
plt.figure()
plt.plot(x[:int(X_size/2)]/del_x, well_slice[:int(X_size/2)]*J2eV, "k")
plt.plot(x[:int(X_size/2)]/del_x, eigentstate, "b")
print(well_slice[:int(X_size/2)]*J2eV)
plt.grid()
plt.ylim(-1, .5)
plt.show()

pim = np.zeros((X_size, Y_size))
prl = np.zeros((X_size, Y_size))

pim_test = np.zeros((X_size, Y_size))
prl_test = np.zeros((X_size, Y_size))

pml = np.ones((X_size, Y_size))
#pml = aabc.abcx2d(pml, 50)
pml = aabc.abcy2d(pml, 50)
#vf.draw_circular_line(eigentstate, prl, Metal_locx, Metal_locy)

V = np.where(V <= 500*eV2J, V, .2*eV2J)
#prl, pim = quantum.normalize(prl, pim)
#fig = plt.figure()
#plt.plot(eigen_values[0:10]*J2eV,'ok')
#plt.show()
#ax = fig.add_subplot(projection='3d')
#surf = ax.plot_surface(X, Y, V*J2eV, rstride =10, cstride = 1, cmap=cm.viridis, linewidth = .4)
#ax.set_zlim(-.2, .2)
plt.show()


#psi, ptot, ke, pe, E = quantum.update_measurables(1, prl, pim, del_x, V*metal)

#print(f"ptot = {ptot}")
#print(f"ke = {ke}")
#print(f"pe = {pe}")
#print((eigen_values[0]*J2eV))







time = sim.fdtdnd(2000, 2, prl_test, pim_test, V_test, ra, rd, t=0, part_id = 1, abc=pml, source=source)
psi_test, ptot_test, ke_test, pe_test, E_test = quantum.update_measurables(2, prl_test, pim_test, del_x, V_test)
p_well_test = quantum.find_ptot_range(prl_test, pim_test, (50, 150))
usr_ip =""
time = 0



fig = plt.figure(figsize=plt.figaspect(1.))
ax0 = fig.add_subplot(1,1,1, projection='3d')
ax0.plot_surface(X*1E9, Y*1E9, prl_test, rstride =10, cstride = 1, cmap=cm.plasma, linewidth = .4)
plt.show()
#print(p_well_test)
E_x = np.linspace(0, .8, 200)
E_y = np.linspace(0, .8, 200)

EX, EY = np.meshgrid(E_x, E_y)
PSI = np.zeros((X_size, Y_size), dtype=np.complex128)

np.save("prl_test.npy", prl_test)
np.save("pim_test.npy", pim_test)
while usr_ip is not "0" :
   usr_ip = input("Steps: ")
   try:
       cleaned_usr_ip = int(usr_ip)
   except ValueError:
       continue
   for _ in range(cleaned_usr_ip):
     time = sim.fdtdnd(1, 2, prl, pim, V, ra, rd, t=time, abc=pml, source=source,
                         misc=misc)
        
     psi, ptot, ke, pe, E = quantum.update_measurables(2, prl, pim, del_x, V*metal)
     PSI += quantum.DFT(prl, pim, time, dt, EX, PSI)
      
   fig = plt.figure(figsize=plt.figaspect(1))
   ax0 = fig.add_subplot(2,2,1)
   ax0.contour(X*1E9, Y*1E9, prl, rstride =10, cstride = 1, cmap=cm.plasma, linewidth = .4)
   ax0.contour(X*1E9, Y*1E9, V*J2eV, rstride =10, cstride = 1, cmap=cm.viridis, linewidth = .4)
     
     
   ax1 = fig.add_subplot(2,2,2, projection='3d')
   ax1.plot_surface(X*1E9, Y*1E9, prl, rstride =10, cstride = 1, cmap=cm.plasma, linewidth = .4)
   #ax1.plot_surface(X*1E9, Y*1E9, V*J2eV, rstride =10, cstride = 1, cmap=cm.viridis, linewidth = .4)
   #ax1.set_zlim(-.01, .01)
   ax2 = fig.add_subplot(2,1,2)
   ax2.plot(X[Metal_locx]*1E9, prl[Metal_locx], color="blue", linestyle="solid", label="prl")
   ax2.plot(X[Metal_locx]*1E9, pim[Metal_locx], color="red", linestyle="dashed", label="pim")
   ax2.plot(X[Metal_locx]*1E9, V[Metal_locx]*J2eV, color="black", linestyle="solid", label="V")
   ax2.legend(loc="upper right")
   ax2.grid()
   ax2.set_ylim(-.01,.01)
   fig.text(0, 0, f"time {time*dt*1E12 : .4f}ps")
   fig.text(.3, 0, f"steps {time}")
   fig.text(0, -.05, f"ke {ke : .2f}eV")
   fig.text(0, -.1, f"pe {pe : .2f}eV")
   fig.text(0, -.15 ,f"E total: {E :.2f}eV")
   fig.text(.30
             , -.15 ,f"likelhood of finding particle: {(ptot/ptot_test)*100 : .2f}%")
     
   p_well = quantum.find_ptot_range(prl, pim, (50, 150))
   plt.tight_layout()
   plt.show()
    
   fig = plt.figure()
   ax0 = fig.add_subplot()
   ax0.contour(EX, EY, np.abs(PSI), linewidth = .4)
   ax0.set_ylim(-.2,.2)
   ax0.set_xlim(-.2,.2)
   plt.show()
    #print(p_well/p_well_test)
    #print(f"ptot = {ptot}")
    #print(f"ke = {ke}")
    #print(f"pe = {pe}")
    #print(f"E = {E}")
    #print(f"t = {time*dt}")

np.save("prl.npy", prl)
np.save("pim.npy", pim)