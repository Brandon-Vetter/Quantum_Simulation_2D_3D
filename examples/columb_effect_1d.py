import sys
sys.path.append("../src")
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
from sim import *
import sim

sigma = .2
lambd = .01*eV2J
X_size = 4000
Y_size = 400
Z_size = 100
del_x = 0.1e-9
dt = 1e-17
ra = (0.5*hbar_J/meff)*(dt/del_x**2)
rd = dt/hbar_J
sim_time = 3.5E-12

@jit 
def source(t, prl, pim, V, ra, rd, part_id):
    if part_id == 0:
        sources.psource1d(t, 600, prl, pim, dt, .02)
    else:
        sources.psource1d(t, 3200, prl, pim, dt, .02)

sim.source = source

x = np.linspace(0, X_size*del_x, X_size)
pml = aabc.abc1d(x, 400)
plt.plot(x, pml)

particles = {}
prl1 = np.zeros(X_size)
pim1 = np.zeros(X_size)
particles[0] = (prl1,pim1)

prl2 = np.zeros(X_size)
pim2 = np.zeros(X_size)
particles[1] = (prl2,pim2)



V = np.zeros(X_size)
user_ip = ""
hf = np.zeros(len(V))
time = 0
animation = input("Animation?")
if animation.Contains("N"):
    while True:
    
        user_ip = int(input("time step: "))
        if user_ip == 0:
            break
        time, hf = fdtdndmp(user_ip, 1, particles, V, ra, rd, del_x, abc=pml,
                            t=time, attract=-1)
        fig, ax = plt.subplots(2,1)
        ax[0].grid()
        ax[1].grid()
        prlp1, = ax[0].plot(x*1E9, prl1, "b", label="P1 real")
        pimp1, = ax[0].plot(x*1E9, pim1, "--r", label="P1 imaginary")
        prlp2, = ax[0].plot(x*1E9, prl2, "orange", label="P2 real")
        pimp2, = ax[0].plot(x*1E9, pim2, "--c", label="P2 imaginary")
        fig.text(.1,.01,f"{time*dt*1E12 : .4f}ps")
        hfp, = ax[1].plot(x*1E9, hf*J2eV, label="coloumb")
        ax[0].legend(loc=(1.01, .2))
        ax[1].set_title("columb energy")
        ax[0].set_title("Particles")
        ax[1].legend(loc=(1.01, .5))
        plt.tight_layout()
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
    time, hf = fdtdndmp(100, 1, particles, V, ra, rd, del_x, abc=pml, t=time,
                        attract=-1)
    
    #print(f"{time*dt*1E12}ps")
    prlp1.set_ydata(prl1)
    pimp1.set_ydata(pim1)
    prlp2.set_ydata(prl2)
    pimp2.set_ydata(pim2)
    hfp.set_ydata(hf*J2eV)
    time_text.set_text(f"{time*dt*1E12 : .4f}ps")
    return time
    


fdtd_update = 100
frame_step = dt*fdtd_update
frames_to_set_time = int(sim_time/frame_step)
ani = animation.FuncAnimation(fig, partial(update), frames=frames_to_set_time, interval=50)
plt.show()
ani.save("coulumb_animation_2.mp4")
    