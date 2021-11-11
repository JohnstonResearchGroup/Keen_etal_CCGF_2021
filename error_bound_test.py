import os
import sys

import numpy as np
import numpy.linalg as npla

import matplotlib.pyplot as plt

from scipy.linalg   import expm
two_site = False
three_site = True
I2 = np.array([[1.0, 0.0],[0.0, 1.0]])
X  = np.array([[0.0, 1.0],[1.0, 0.0]])
Y  = np.array([[0.0,-1.j],[1.j, 0.0]])
Z  = np.array([[1.0, 0.0],[0.0,-1.0]])

Z1 = np.kron(np.kron(np.kron( Z,I2),np.kron(I2,I2)),np.kron(I2,I2))
Z2 = np.kron(np.kron(np.kron(I2, Z),np.kron(I2,I2)),np.kron(I2,I2))
Z3 = np.kron(np.kron(np.kron(I2,I2),np.kron( Z,I2)),np.kron(I2,I2))
Z4 = np.kron(np.kron(np.kron(I2,I2),np.kron(I2, Z)),np.kron(I2,I2))
Z5 = np.kron(np.kron(np.kron(I2,I2),np.kron(I2,I2)),np.kron( Z,I2))
Z6 = np.kron(np.kron(np.kron(I2,I2),np.kron(I2,I2)),np.kron(I2, Z))

X1 = np.kron(np.kron(np.kron(X,I2), np.kron(I2,I2)),np.kron(I2,I2))
X2 = np.kron(np.kron(np.kron(Z, X), np.kron(I2,I2)),np.kron(I2,I2))
X3 = np.kron(np.kron(np.kron(Z, Z), np.kron(X, I2)),np.kron(I2,I2))
X4 = np.kron(np.kron(np.kron(Z, Z), np.kron(Z,  X)),np.kron(I2,I2))
X5 = np.kron(np.kron(np.kron(Z, Z), np.kron(Z,  Z)),np.kron( X,I2))
X6 = np.kron(np.kron(np.kron(Z, Z), np.kron(Z,  Z)),np.kron( Z, X))

Y1 = np.kron(np.kron(np.kron(Y,I2), np.kron(I2,I2)),np.kron(I2,I2))
Y2 = np.kron(np.kron(np.kron(Z, Y), np.kron(I2,I2)),np.kron(I2,I2))
Y3 = np.kron(np.kron(np.kron(Z, Z), np.kron(Y, I2)),np.kron(I2,I2))
Y4 = np.kron(np.kron(np.kron(Z, Z), np.kron(Z,  Y)),np.kron(I2,I2))
Y5 = np.kron(np.kron(np.kron(Z, Z), np.kron(Z,  Z)),np.kron( Y,I2))
Y6 = np.kron(np.kron(np.kron(Z, Z), np.kron(Z,  Z)),np.kron( Z, Y))

X1X2 = np.kron(np.kron(np.kron(X,X), np.kron(I2,I2)),np.kron(I2,I2))
X1Z2X3 = np.kron(np.kron(np.kron(X,Z), np.kron(X,I2)),np.kron(I2,I2))
Y1Y2 = np.kron(np.kron(np.kron(Y,Y), np.kron(I2,I2)),np.kron(I2,I2))
Y1Z2Y3 = np.kron(np.kron(np.kron(Y,Z), np.kron(Y,I2)),np.kron(I2,I2))

X4X5 = np.kron(np.kron(np.kron(I2,I2), np.kron(I2,X)),np.kron(X,I2))
X4Z5X6 = np.kron(np.kron(np.kron(Y,I2), np.kron(I2,X)),np.kron(Z,X))
Y4Y5 = np.kron(np.kron(np.kron(Y,I2), np.kron(I2,Y)),np.kron(Y,I2))
Y4Z5Y6 = np.kron(np.kron(np.kron(Y,I2), np.kron(I2,Y)),np.kron(Z,Y))

am = 0.5 * (X + 1.j * Y)
ap = 0.5 * (X - 1.j * Y)

a1p = np.kron(np.kron(np.kron(ap,I2), np.kron(I2,I2)),np.kron(I2,I2))
a2p = np.kron(np.kron(np.kron( Z,ap), np.kron(I2,I2)),np.kron(I2,I2))
a3p = np.kron(np.kron(np.kron( Z, Z), np.kron(ap,I2)),np.kron(I2,I2))
a4p = np.kron(np.kron(np.kron( Z, Z), np.kron( Z,ap)),np.kron(I2,I2))
a5p = np.kron(np.kron(np.kron( Z, Z), np.kron( Z, Z)),np.kron(ap,I2))
a6p = np.kron(np.kron(np.kron( Z, Z), np.kron( Z, Z)),np.kron( Z,ap))

a1 = np.conj(a1p).T
a2 = np.conj(a2p).T
a3 = np.conj(a3p).T
a4 = np.conj(a4p).T
a5 = np.conj(a5p).T
a6 = np.conj(a6p).T

Nt   = 601
T0   = 0
T1   = 18
Time = np.linspace(T0,T1,Nt)
Ttot = T1 - T0
dT   = Ttot/(Nt-1)
print(dT)
'''
parameters, trial, and Hamiltonian
'''
U = 8
mu = U/2
e0 = 0

##########################
# init DMFT parameter #
##########################
e1 = -0.39345870615292378
e2 = 0.39345890659623484
V1 = 0.62681957596523907
V2 = 0.62681965580327426

H1 = (e0-mu)*a1p@a1 + (e1-mu)*a2p@a2 + (e1-mu)*a5p@a5
H1 += (e0-mu)*a4p@a4 + (e2-mu)*a3p@a3 + (e2-mu)*a6p@a6

H2 = V1*(a1p@a2 + a2p@a1 + a5p@a4 + a4p@a5)
H2 += V2*(a3p@a1 + a1p@a3 + a4p@a6 + a6p@a4)
H1 += U*a1p@a1@a4p@a4

HX1X2 = V1*X1X2
HX1Z2X3 = V2*X1Z2X3
HY1Y2 = V1*Y1Y2
HY1Z2Y3 = V2*Y1Z2Y3

HX4X5 = V1*X4X5
HX4Z5X6 = V2*X4Z5X6
HY4Y5 = V1*Y4Y5
HY4Z5Y6 = V2*Y4Z5Y6

H = H1 + H2

def Utrot(dt, n_steps, h1, h2):
    return npla.matrix_power(expm(-1j*2*np.pi*dt*0.5*h1)@expm(-1j*2*np.pi*dt*h2)@expm(-1j*2*np.pi*dt*0.5*h1), n_steps)

up_bound_trot_err3 = {}
trot_err3 = {}

com13 = H2@H1 - H1@H2
nest_comBBA3 = H2@com13 - com13@H2
nest_comAAB3 = H1@(-1*com13) - (-1*com13)@H1

normAB3 = npla.norm(com13,2)

normBBA3 = npla.norm(nest_comBBA3,2)
normAAB3 = npla.norm(nest_comAAB3,2)
terms = [normBBA3, normAAB3]
print('Commutator Norms: ', terms, '\n')
for i in range(2, 6):
    trot_steps=2**i
    delta_t = dT/trot_steps

    trot_error = np.zeros(Nt)
    tol = 1e-16
    up_bound_trot_err3.update({trot_steps: np.zeros(Nt)})
    trot_err3.update({trot_steps: trot_error})

    up_bnd = trot_steps * (2*np.pi)**3 * (dT/trot_steps)**3 * (terms[0]/(12.0)  + terms[1]/(24.0))

    for j in range(0,Nt):
        t = Time[j]
        Ur = expm(-1j*2*np.pi*H*t)
        Urtrot = Utrot(delta_t, trot_steps*j, H1, H2)
        Udiff = Ur - Urtrot
        trot_err3[trot_steps][j] = npla.norm(Udiff, 2)
        up_bound_trot_err3[trot_steps][j] = j*up_bnd
markers = ['bv', 'g^', 'm<','r>']

plot_time3 = Time[0::10]
plot_trot_err43 = trot_err3[4][0::10]
plot_trot_err83 = trot_err3[8][0::10]
plot_trot_err163 = trot_err3[16][0::10]
plot_trot_err323 = trot_err3[32][0::10]

up_bnd43 = up_bound_trot_err3[4][0::10]
up_bnd83 = up_bound_trot_err3[8][0::10]
up_bnd163 = up_bound_trot_err3[16][0::10]
up_bnd323 = up_bound_trot_err3[32][0::10]





##########################
# Two site DMFT parameter #
##########################
e1 = U/2
#e2 = 6.0770501876063845
V1 = 1.0
#V2 = 0.14931472606779114

H1 = (e0-mu)*a1p@a1 + (e1-mu)*a2p@a2
H1 += (e0-mu)*a3p@a3 + (e1-mu)*a4p@a4

H2 = V1*(a1p@a2 + a2p@a1 + a3p@a4 + a4p@a3)
#H2 += V2*(a3p@a1 + a1p@a3 + a4p@a6 + a6p@a4)
H1 += U*a1p@a1@a3p@a3

#HX1X2 = V1*X1X2
#HX1Z2X3 = V2*X1Z2X3
#HY1Y2 = V1*Y1Y2
#HY1Z2Y3 = V2*Y1Z2Y3

#HX4X5 = V1*X4X5
#HX4Z5X6 = V2*X4Z5X6
#HY4Y5 = V1*Y4Y5
#HY4Z5Y6 = V2*Y4Z5Y6

H = H1 + H2

def Utrot(dt, n_steps, h1, h2):
    return npla.matrix_power(expm(-1j*2*np.pi*dt*0.5*h1)@expm(-1j*2*np.pi*dt*h2)@expm(-1j*2*np.pi*dt*0.5*h1), n_steps)

up_bound_trot_err = {}
trot_err = {}

com1 = H2@H1 - H1@H2
nest_comBBA = H2@com1 - com1@H2
nest_comAAB = H1@(-1*com1) - (-1*com1)@H1

normAB = npla.norm(com1,2)

normBBA = npla.norm(nest_comBBA,2)
normAAB = npla.norm(nest_comAAB,2)
terms = [normBBA, normAAB]
print('Commutator Norms: ', terms, '\n')
for i in range(2, 6):
    trot_steps=2**i
    delta_t = dT/trot_steps

    trot_error = np.zeros(Nt)
    tol = 1e-16
    up_bound_trot_err.update({trot_steps: np.zeros(Nt)})
    trot_err.update({trot_steps: trot_error})

    up_bnd = trot_steps * (2*np.pi)**3 * (dT/trot_steps)**3 * (terms[0]/(12.0)  + terms[1]/(24.0))

    for j in range(0,Nt):
        t = Time[j]
        Ur = expm(-1j*2*np.pi*H*t)
        Urtrot = Utrot(delta_t, trot_steps*j, H1, H2)
        Udiff = Ur - Urtrot
        trot_err[trot_steps][j] = npla.norm(Udiff, 2)
        up_bound_trot_err[trot_steps][j] = j*up_bnd
markers = ['bv', 'g^', 'm<','r>']

plot_time = Time[0::10]
plot_trot_err4 = trot_err[4][0::10]
plot_trot_err8 = trot_err[8][0::10]
plot_trot_err16 = trot_err[16][0::10]
plot_trot_err32 = trot_err[32][0::10]

up_bnd4 = up_bound_trot_err[4][0::10]
up_bnd8 = up_bound_trot_err[8][0::10]
up_bnd16 = up_bound_trot_err[16][0::10]
up_bnd32 = up_bound_trot_err[32][0::10]
xlabels=[]
ticks = np.arange(2,18)
x=np.arange(2,18)
for i in range(0,len(ticks)):

    if i%2==0:
        xlabels.append(int(x[i]))

    else:
        xlabels.append('')

#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.family"] = "Helvetica"
#plt.rc('font', family='Helvetica',size=11,weight='normal')
fig, axs = plt.subplots(1, 2, dpi=200, sharex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=11,weight='normal')
axs[0].set_xlabel('Time [1/t*]', fontweight='normal',fontsize=11)
axs[1].set_xlabel('Time [1/t*]', fontweight='normal',fontsize=11)
#plt.xticks(ticks, labels=xlabels)
plt.xlim(4,18)
#plt.ylim(6.5,6.7)
axs[0].set_ylabel('$\dfrac{\mathrm{Upper \ Bound}}{||\mathrm{U(t)} - \mathrm{U_{\mathrm{trot}}(t)}||}$', fontweight='normal',fontsize=11)
axs[0].plot(plot_time, up_bnd4/plot_trot_err4 ,markers[0], markersize=3.0,label='4 steps')
axs[0].plot(plot_time, up_bnd8/plot_trot_err8, markers[1], markersize=3.0,label='8 steps')
axs[0].plot(plot_time, up_bnd16/plot_trot_err16, markers[2], markersize=3.0,label='16 steps')
axs[0].plot(plot_time, up_bnd32/plot_trot_err32 , markers[3], markersize=3.0,label='32 steps')


#axs[0].ylabel('$\dfrac{\mathrm{Upper \ Bound}}{||\mathrm{U(t)} - \mathrm{U_{\mathrm{trot}}(t)}||}$', fontweight='normal',fontsize=11)
axs[1].plot(plot_time, up_bnd43/plot_trot_err43 ,markers[0], markersize=3.0,label='4 steps')
axs[1].plot(plot_time, up_bnd83/plot_trot_err83, markers[1], markersize=3.0,label='8 steps')
axs[1].plot(plot_time, up_bnd163/plot_trot_err163, markers[2], markersize=3.0,label='16 steps')
axs[1].plot(plot_time, up_bnd323/plot_trot_err323 , markers[3], markersize=3.0,label='32 steps')
axs[0].annotate("a", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
axs[1].annotate("b", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
#plt.plot(Time.real, up_bound_trot_err[4].real, 'b', markersize=2.0)
#plt.plot(Time.real, up_bound_trot_err[8].real, 'g', markersize=2.0)
#plt.plot(Time.real, up_bound_trot_err[16].real, 'm',markersize=2.0)
#plt.plot(Time.real, up_bound_trot_err[32].real, 'r', markersize=2.0)
axs[0].legend(frameon=False)
plt.tight_layout(pad=0.5)
fig.savefig('trot_err2.pdf',dpi=None, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()
