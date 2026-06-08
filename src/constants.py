"""
Physical and simulation constants used throughout the project.

:author: Brandon Vetter
:date: 4/12/26


Constants
---------
- ``hbar_J``: reduced Planck constant in J*s
- ``hbar_eV``: reduced Planck constant in eV*s
- ``h_nobar_eV``: Planck constant (not reduced) in eV*s
- ``h_nobar_J``: Planck constant (not reduced) in J*s
- ``m0``: electron rest mass in kg
- ``meff``: effective mass used in simulations (defaults to ``m0``)
- ``ecoul``: elementary charge in Coulombs
- ``epsz``: vacuum permittivity (in SI units)
- ``eV2J``: factor to convert eV -> Joules
- ``J2eV``: factor to convert Joules -> eV
- ``G0``: quantum conductance in microSiemens

Semiconductor effective masses (relative to ``m0``)
 - ``Si``: silicon effective mass (m*/m0)
 - ``GaAs``: gallium-arsenide effective mass (m*/m0)
 - ``Ge``: germanium effective mass (m*/m0)

 
 Property of the University of Idaho
"""


######################
# Global Varables
#####################

# Quantum Varables

hbar_J = 1.054e-34 # J*s
hbar_eV = 6.58e-16 # eV
h_nobar_eV = 4.135e-15  # in eV
h_nobar_J = 6.625e-34 # J*s
m0 = 9.1e-31 # kg
meff = m0 
ecoul = 1.6e-19 # C
epsz = 8.85e-9 # H/m permebilit of free space
eV2J = 1.6e-19 # Convert to Joules
J2eV = 1/eV2J # Convert to eV
G0 = 38.7 # micro S Quantum Conductance


# Semiconductor Values
    # all in m/me
Si = 1.08
GaAs = 0.067
Ge = 0.55

# common varables
eV = 1.602E-19
q = 1.602E-19
k_J = 1.38E-23
k_eV = k_J/eV
perm = 8.85E-12 # m
c = 3E8
thermal_V_at_300 = (k_J*300)/q

# silicion
si_mn = .26*m0
si_mp = .39*m0
si_Eg = 1.12
si_un = 1417
si_up = 471
si_lattice_cosnt = 5.43
si_e_af = 4.05

# dieletrics
si_di = 11.8*perm
siO2_di = 3.9*perm
