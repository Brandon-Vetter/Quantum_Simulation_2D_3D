import sys
sys.path.append("../src")

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import sources
from sim import *
import sim
from constants import *

X_size = 1500
Y_size = 400
Z_size = 100
del_x = 0.2e-9

dt = .5e-16
ra = (0.5*hbar_J/meff)*(dt/del_x**2)
rd = dt/hbar_J
sigma = .5
lambd = .01*eV2J



@jit 
def source(t, prl, pim, V, ra, rd, part_id):
    sources.lysource2d(t, 300, prl, pim, dt, .3)

sim.source = source
ra = (0.5*hbar_J/meff)*(dt/del_x**2)
rd = dt/hbar_J

x = np.linspace(0, X_size*del_x, X_size)
y = np.linspace(0, Y_size*del_x, Y_size)
# z = np.linspace(0, Z_size*del_x, Z_size)

X, Y = np.meshgrid(x, y)

pml = aabc.abcy2d(X, 250)

prl = np.zeros((Y_size,X_size))
pim = np.zeros((Y_size,X_size))
V = np.zeros((Y_size,X_size))
vf.sten(V, 1)
vf.draw_circular_coulmbc(V, 200, 800, 10000, -1)
vf.draw_cylender(V, 200, 800, 80, 1, force=True)
time = 0

user_ip = ""
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(X*1E9, Y*1E9, V*J2eV, rstride =10, cstride = 1, cmap=cm.plasma, linewidth = .4)
while user_ip != "0":
    user_ip = input("time step: ")
    
    try:
        time = fdtdnd(int(user_ip), 2, prl, pim, V, ra, rd, t=time, abc=pml)
    except ValueError:
        continue
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    psi_2 = (prl + 1j*pim)**2
    surf = ax.plot_surface(X*1E9, Y*1E9, psi_2, rstride =10, cstride = 1, cmap=cm.plasma, linewidth = .4)
    #    ax.view_init(elev=30., azim = 30)
    ax.view_init(elev=30., azim = -75)
        #   plt.text(10,10,.2,"T = : {}".format(round(T,0)) )
    plt.ylabel('Y (nm)')
        #   ax.text2D(5,5,"T = 0")
    plt.xlabel('X (nm)')    

    print(f"time: {time*dt*1E12 : .2f} ps")
    plt.savefig("3dexample.png")
    plt.show()