import os
import sys
import math, cmath
import re

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import cosm, expm, sinm
from numpy.linalg import matrix_power
from scipy.optimize import (curve_fit, minimize)
from Cluster_ops import *
from Pauli_matrices import *
from Params import *
import csv

duration = 1  # seconds
freq = 440  # Hz

V = float(sys.argv[1])

print('mu1=', mu1, 'mu2=', mu2, 'nu1=', nu1, 'nu2=', nu2)

sigma_sing_ex = lam12_val(U,V)*(c2dag@c1 - c1dag@c2 + c3dag@c4 - c4dag@c3)
sigma_doub_ex = (lamdoub_val(U,V) - lam12_val(U,V)**2).real*(c2dag@c3dag@c4@c1 - c1dag@c4dag@c3@c2)

H_SIAM =  U/4.0 * (Z1Z3 - Z1 - Z3 + I) + mu/2.0 * (Z1 + Z3) \
         +V/2.0 * (X1X2 + Y1Y2 + X3X4 + Y3Y4) - U/2.0 * I

Hu = U/4.0 * Z1Z3 - U/4.0 * I
Hv12 = V/2.0 * (X1X2 + Y1Y2)
Hv34 = V/2.0 * (X3X4 + Y3Y4)
Hv = Hv12+Hv34
H_mu = mu/2 * (Z1 + Z3)

t21 = t21_val(U,V)

def E0N(hamiltonian, trial_vec, u, v):
    return (np.conjugate(trial_vec).transpose()@expm(-T_op(u,v))@hamiltonian@expm(T_op(u,v))@trial_vec)

if (abs(V-1.0) < 1.0e-2):
    A = 0.657239167560257 # (1 - eTau)/(1 - eSig*eTau)
    B = -0.1237935177229450
elif (abs(V-0.01) < 1.0e-5):
    A = 0.7003
    B = -0.1145
else:
    raise ValueError('Unknown A and B for these parameters!')

Ecc = E0N(H_SIAM, trial_state, U, V)
print('Ecc= ', Ecc)
iHI = -1j*Ecc*I
iH  = 1j*(H_SIAM - Ecc*I)

dUr_trot = expm(2.0*np.pi*complex(0.0, -1.0)*Hu*0.5*dt) @ expm(2.0*np.pi*complex(0.0, -1.0)*Hv*dt) @ expm(2.0*np.pi*complex(0.0, -1.0)*Hu*0.5*dt)@expm(-2.0*np.pi*iHI*dt)
dUa_trot = expm(2.0*np.pi*complex(0.0, 1.0)*Hu*0.5*dt) @ expm(2.0*np.pi*complex(0.0, 1.0)*Hv*dt) @ expm(2.0*np.pi*complex(0.0, 1.0)*Hu*0.5*dt)@expm(2.0*np.pi*iHI*dt)

X3UrX3=np.zeros(steps+1,dtype=complex)
X3UrX1X2X3=np.zeros(steps+1,dtype=complex)
X1X2X3UrX3=np.zeros(steps+1,dtype=complex)
X1X2X3UrX1X2X3=np.zeros(steps+1,dtype=complex)

X3Ur_trotX3=np.zeros(steps+1,dtype=complex)
X3Ur_trotX1X2X3=np.zeros(steps+1,dtype=complex)
X1X2X3Ur_trotX3=np.zeros(steps+1,dtype=complex)
X1X2X3Ur_trotX1X2X3=np.zeros(steps+1,dtype=complex)

G_impt=np.zeros(steps+1,dtype=complex)
G_impt_trot=np.zeros(steps+1,dtype=complex)

Ur_expecval = np.zeros(steps+1, dtype=complex)

for i in np.arange(0, steps+1):
    print('Time step ', i, ' of ', steps)
    t = time_arr[i]
    Ur = expm(-2*np.pi*1j*(H_SIAM - Ecc*I)*t)
    Ua = expm(+2*np.pi*1j*(H_SIAM + Ecc*I)*t)
    #if i !=0:
    Ur_trot = matrix_power(dUr_trot, i)
    Ua_trot = matrix_power(dUa_trot, i)

    Ur_expecval[i] = trial_state_dag@Ur_trot@trial_state

################################################################################
#                           Exact Unitaries                                    #
################################################################################

    X3UrX3[i] = trial_state_dag@X3@Ur@X3@trial_state
    X3UrX1X2X3[i] = trial_state_dag@X3@Ur@X1@X2@X3@trial_state
    X1X2X3UrX1X2X3[i] = trial_state_dag@X1@X2@X3@Ur@X1@X2@X3@trial_state

################################################################################
#                           Trotterized Unitaries                              #
################################################################################

    X3Ur_trotX3[i] = trial_state_dag@X3@Ur_trot@X3@trial_state
    X3Ur_trotX1X2X3[i] = trial_state_dag@X3@Ur_trot@X1@X2@X3@trial_state
    X1X2X3Ur_trotX1X2X3[i] = trial_state_dag@X1@X2@X3@Ur_trot@X1@X2@X3@trial_state

    G_impt_trot[i] = nu1*mu1*X3Ur_trotX3[i]+nu1*mu2*X3Ur_trotX1X2X3[i]+nu2*mu2*X1X2X3Ur_trotX1X2X3[i]
    G_impt[i] = nu1*mu1*X3UrX3[i]+nu1*mu2*X3UrX1X2X3[i]+nu2*mu2*X1X2X3UrX1X2X3[i]
#print(sigX1Ua_trotX1)
#print(len(sigX3UrX3))
#print(Ga_imptA)
print('G_impt_trot: ', G_impt_trot.imag, '\n')
print('X3Ur_trotX3: ', X3Ur_trotX3.imag, '\n')
print('X3Ur_trotX1X2X3: ', X3Ur_trotX1X2X3.imag, '\n')
print('X1X2X3Ur_trotX3: ', X1X2X3Ur_trotX3.imag, '\n')
print('X1X2X3Ur_trotX1X2X3: ', X1X2X3Ur_trotX1X2X3.imag, '\n')

output = np.column_stack((time_arr.real,G_impt,\
X3UrX3, X3UrX1X2X3, X1X2X3UrX1X2X3))
np.savetxt('Time_Domain_Data/Classical/newCC_G_impt_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V, steps,total_time),output, header="time, G_impt, X3UrX3, X3UrX1X2X3, X1X2X3UrX1X2X3", delimiter='\t')


output = np.column_stack(((time_arr.real).flatten(),G_impt_trot.flatten(),\
X3Ur_trotX3.flatten(), X3Ur_trotX1X2X3.flatten(), X1X2X3Ur_trotX1X2X3.flatten()))
np.savetxt('Time_Domain_Data/Classical/newCC_G_impt_trot_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V, steps,total_time),output, header="time, G_impt_trot, X3Ur_trotX3, X3Ur_trotX1X2X3, X1X2X3Ur_trotX1X2X3", delimiter='\t')
print('<Ur>: ', Ur_expecval)

f=plt.figure(1)
plt.plot(time_arr, G_impt.imag, label='Exact')
plt.plot(time_arr, G_impt_trot.imag, label='Trotter')
plt.legend()

g=plt.figure(2)
plt.plot(time_arr, Ur_expecval.real, label='<Ur>')
plt.legend()

plt.show()

#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
