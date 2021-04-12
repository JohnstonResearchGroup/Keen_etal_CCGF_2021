import os
import sys

import numpy as np
import numpy.linalg as npla

import matplotlib.pyplot as plt

from scipy.linalg   import expm

approx = False

'''
Define Pauli matrices
YZ =  iX
ZY = -iX
XZ = -iY
ZX =  iY
'''
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

I  = np.eye(64)

'''
trial vector
'''
ket0  = np.array([1.,0.])
ket1  = np.array([0.,1.])

'''
parameters, trial, and Hamiltonian
'''
U = 8
mu = U/2
e0 = 0

init = sys.argv[1]== 'True'
if (init == False):
#    ##########################
#    # DMFT parameter #
#    ##########################
    e1 = -4.1247732492596239
    e2 = 6.0770501876063845
    V1 = 0.99485639568316897
    V2 = 0.14931472606779114
    t31 =  -0.02521902157743792
    t32 =  -0.002465255916962233
    t65 =  0.0011842247635256034
    t45 =  -0.08154020427583772
    t3615 =  -1.449244105004498e-05
    t3625 =  -1.2127862868433207e-06
    t3415 =  0.0015931165094268355
    t3425 =  7.044612627328341e-05
    l13 =  -0.02533158906629098
    l23 =  -0.0024693556891348463
    l56 =  0.001176756596395727
    l54 =  -0.0810408514515095
    l1536 =  -4.403572068599451e-05
    l2536 =  -4.10223381930596e-06
    l1534 =  0.00362301211926025
    l2534 =  0.0002694947617477684
else:

#    ##################
#    # DMFT initial parameter #
#    ##################
    e1 = -0.39345870615292378
    e2 = 0.39345890659623484
    V1 = 0.62681957596523907
    V2 = 0.62681965580327426
    t31 =  -1.130297642351575
    t32 =  -0.48111981471270887
    t65 =  0.16454558498519073
    t45 =  -0.1236106345855691
    t3615 =  -0.10330157953281678
    t3625 =  -0.018124569391270252
    t3415 =  0.5059484094981708
    t3425 =  0.01751499188476848
    l13 =  -0.4100289574935819
    l23 =  -0.16517247243540775
    l56 =  0.17550367292542507
    l54 =  -0.29029587473610774
    l1536 =  -0.09431113943907284
    l2536 =  -0.03171795643239405
    l1534 =  0.21049486215438304
    l2534 =  0.02509855455555259
    '''
    e1 = 0.0
    e2 = 0.0
    V1 = 0.0
    V2 = 0.0
    t31 =  0.0
    t32 =  0.0
    t65 =  0.0
    t45 =  0.0
    t3615 =  0.0
    t3625 =  0.0
    t3415 =  0.0
    t3425 =  0.0
    l13 =  0.0
    l23 =  0.0
    l56 =  0.0
    l54 =  0.0
    l1536 =  0.0
    l2536 =  0.0
    l1534 =  0.0
    l2534 =  0.0
    '''
H1 = (e0-mu)*a1p@a1 + (e1-mu)*a2p@a2 + (e1-mu)*a5p@a5
H1 += (e0-mu)*a4p@a4 + (e2-mu)*a3p@a3 + (e2-mu)*a6p@a6

H2 = V1*(a1p@a2 + a2p@a1 + a5p@a4 + a4p@a5)
H2 += V2*(a3p@a1 + a1p@a3 + a4p@a6 + a6p@a4)
H1 += U*a1p@a1@a4p@a4

H = H1 + H2

trial = np.kron(np.kron(np.kron(ket1, ket1),np.kron(ket0, ket0)),np.kron(ket1,ket0))

T1  = t31*a3p@a1 + t32*a3p@a2 + t65*a6p@a5 + t45*a4p@a5
T2  = t3615*a3p@a6p@a5@a1 + t3625*a3p@a6p@a5@a2\
     +t3415*a3p@a4p@a5@a1 + t3425*a3p@a4p@a5@a2
L1  = l13*a1p@a3 + l23*a2p@a3 + l56*a5p@a6 + l54*a5p@a4
L2  = l1536*a1p@a5p@a6@a3 + l2536*a2p@a5p@a6@a3\
     +l1534*a1p@a5p@a4@a3 + l2534*a2p@a5p@a4@a3


def Utrott(t, dt, n_steps, h_1, h_2):
    #factor_1 = expm(-2.0*np.pi*1j*0.5*dt*h_1)
    #return np.linalg.matrix_power(factor_1@expm(2.0*np.pi*1j*dt*Ecc*I)@expm(-2.0*np.pi*1j*dt*h_2)@factor_1,n_steps)
    return np.linalg.matrix_power(expm(-2.0*np.pi*1j*dt*h_1)@expm(2.0*np.pi*1j*dt*Ecc*I)@expm(-2.0*np.pi*1j*dt*h_2), n_steps)

T   = T1 + T2
L   = L1 + L2
epT = expm(T)
emT = expm(-T)

Ehf = (np.conj(trial).T@H@trial).real
Ecc = (np.conj(trial).T@emT@H@epT@trial).real
print('Ehf = ',Ehf)
print('Ecc = ',Ecc)
Ecc = (np.conj(trial).T@(I+L)@emT@H@epT@trial).real
print('Ecc = ',Ecc)

'''
#Unitary basis
'''
dims = H.shape[0]
nbas = 6
if approx==True:
    UM = np.zeros((nbas,dims,dims),dtype=float)
    UM[0] = X1
    #UM[1] = X3@X2@X1
    #UM[2] = X6@X5@X1
    UM[3] = X4@X5@X1
    #UM[4] = X3@X6@X5@X2@X1
    #UM[5] = X3@X4@X5@X2@X1
    P  = np.zeros((nbas,dims),dtype=float)
    for i in range(0,nbas):
        P[i] = UM[i]@trial
else:
    UM = np.zeros((nbas,dims,dims),dtype=float)
    UM[0] = X1
    UM[1] = X3@X2@X1
    UM[2] = X6@X5@X1
    UM[3] = X4@X5@X1
    UM[4] = X3@X6@X5@X2@X1
    UM[5] = X3@X4@X5@X2@X1
    P  = np.zeros((nbas,dims),dtype=float)
    for i in range(0,nbas):
        P[i] = UM[i]@trial

'''
All mup/muq, nup/nuq will be given from classcial computing
scalars for "a_p exp(T) |0>" and "a_p^\dagger exp(T) |0>"
'''
if approx==True:
    mup = np.zeros(nbas,dtype=float)
    mup[0] =  1.0
    #mup[1] =  t32
    #mup[2] =  t65
    mup[1] =  t45
    #mup[4] =  t3625 - t32*t65
    #mup[5] =  t3425 - t32*t45
else:
    mup = np.zeros(nbas,dtype=float)
    mup[0] =  1.0
    mup[1] =  t32
    mup[2] =  t65
    mup[1] =  t45
    mup[4] =  t3625 - t32*t65
    mup[5] =  t3425 - t32*t45

'''
scalars for "<0| (1+\Lambda) exp(-T) a_p^\dagger"
and "<0| (1+\Lambda) exp(-T) a_p"
'''
nup = np.zeros(nbas,dtype=float)
nup[0] = (np.conj(trial).T@(I+L)@emT@trial).real
#nup[1] = (np.conj(trial).T@(I+L)@emT@X3@X2@trial).real
#nup[2] = (np.conj(trial).T@(I+L)@emT@X6@X5@trial).real
nup[1] = (np.conj(trial).T@(I+L)@emT@X4@X5@trial).real
#nup[4] = (np.conj(trial).T@(I+L)@emT@X3@X6@X5@X2@trial).real
#nup[5] = (np.conj(trial).T@(I+L)@emT@X3@X4@X5@X2@trial).real

'''
Define time domain and frequency domain
'''
Nt   = 601
T0   = 0
T1   = 18
Time = np.linspace(T0,T1,Nt)
Ttot = T1 - T0
dT   = Ttot/(Nt-1)

Freq = np.linspace(-0.5*Nt/Ttot,0.5*Nt/Ttot,Nt)
dF   = (Nt/(Nt-1)) * (1/Ttot)
print('dT = ',dT,', dF = ',dF)

'''
Spectral function in time domain
'''
iH  = 1j*(H - Ecc*I)
SF_t = np.zeros(Nt,dtype=float)
SF_t_Trot = np.zeros(Nt,dtype=float)


for i in range(0,Nt):
    t = Time[i]
    Ur = expm(-2*np.pi*iH*t)
#    Ua = expm(+2*np.pi*iH*t)
    SF_t[i] = 0.0
    for k in range(0,nbas):
        for l in range(0,nbas):
            # print(np.conj(P[k]).T@Ur@P[l])
            SF_t[i] += nup[k]*mup[l]*(np.conj(P[k]).T@Ur@P[l]).imag
            # SF_t[i] += nu[k]*mu[l]*(np.conj(P[k]).T@Ur@P[l]-np.conj(P[k]).T@Ua@P[l]).imag
#    print(i,SF_t[i])
output4 = np.column_stack((Time, SF_t))
if (init==True):
    np.savetxt('Data/init_ex_G_impt_U=%.2f_tot_time=%.2f.txt' % (U,T1),output4, delimiter='\t')
else:
    np.savetxt('Data/ex_G_impt_U=%.2f_tot_time=%.2f.txt' % (U, T1),output4, delimiter='\t')

'''
f = plt.figure(1)
plt.plot(Time, SF_t, 'b--')
plt.show()
'''
for i in range(2, 6):
    trot_steps=2**i
    delta_t = dT/trot_steps

    trot_error = np.zeros(Nt)
    tol = 1e-16
    for j in range(0,Nt):
        t = Time[j]
        Ur = expm(-2*np.pi*iH*t)
        Urtrot = Utrott(t, delta_t, trot_steps*j, H1, H2)
        Udiff = Ur - Urtrot
        trot_error[j] = abs(npla.norm(Udiff, 2))
        '''
        if j==50:
            #pass
            print('At time ', t)
            print('||Ur|| = ', npla.norm(Ur,2), '\n')
            print('||Urtrot|| = ', npla.norm(Urtrot,2), '\n')
            print('Phase term: ', expm(2*np.pi*t*1j*Ecc*I))
            print('Ur - Urtrot = ', Ur - Urtrot, '\n')
            print('Ur = ', Ur, '\n')
            print('Urtrot = ', Urtrot, '\n')
        else:
            pass
        '''
    #    Ua = Utrott(t, delta_t, -H_1, -H_2)*np.exp(-2.0*np.pi*1j*Ecc)
        SF_t_Trot[j] = 0.0
        for k in range(0,nbas):
            for l in range(0,nbas):
                # print(np.conj(P[k]).T@Ur@P[l])
                SF_t_Trot[j] += nup[k]*mup[l]*(np.conj(P[k]).T@Urtrot@P[l]).imag
                # SF_t[i] += nu[k]*mu[l]*(np.conj(P[k]).T@Ur@P[l]-np.conj(P[k]).T@Ua@P[l]).imag
    #    print(i,SF_t_Trot[i])
    #print(trot_error)
    #print('SF_t - SF_t_Trot:', SF_t-SF_t_Trot)

    output = np.column_stack((Time, SF_t_Trot))
    output3 = np.column_stack((Time, trot_error))

    if (init==True):
        np.savetxt('Data/init_G_impt_U=%.2f_%d_trot_steps_tot_time=%.2f.txt' % (U, trot_steps,T1),output, delimiter='\t')
    else:
        np.savetxt('Data/G_impt_U=%.2f_%d_trot_steps_tot_time=%.2f.txt' % (U,trot_steps,T1),output, delimiter='\t')
#    np.savetxt('Data/trot_error_U=%.2f_V1=%.2f_%d_trot_steps_tot_time=%.2f.txt' % (U,V1,trot_steps,T1),output3, delimiter='\t')


'''
Spectral function in freq domain
'''
ih    = (H - Ecc*I)
SF_f  = np.zeros(Nt,dtype=float)
eta   = 0.1
count0 = 0.0
for i in range(0,Nt):
    Ur = npla.inv((Freq[i] + 1j*eta)*I + ih)
    SF_f[i] = 0.0
    for k in range(0,nbas):
        for l in range(0,nbas):
            SF_f[i] += nup[k]*mup[l]*(np.conj(P[k]).T@Ur@P[l]).imag
    count0 += SF_f[i]*dF

output2 = np.column_stack((Freq,SF_f))
if (init == True):
    np.savetxt('Data/init_ex_G_impw_U=%.2f_tot_time=%.2f.txt' % (U,T1),output2, delimiter='\t')
else:
    np.savetxt('Data/ex_G_impw_U=%.2f_tot_time=%.2f.txt' % (U,T1),output2, delimiter='\t')


'''
Fourier transform SF_t
'''
SF_t_fft = 2*np.fft.fftshift(np.fft.fft(SF_t))/Nt

# find peaks
Amp = []
Ind = []
for i in range(1,int(Nt/2)+1):
    tmp = abs(SF_t_fft[i])
    if tmp > 0.01 and tmp > abs(SF_t_fft[i+1]) and tmp > abs(SF_t_fft[i-1]):
        Ind.append(i)
        Amp.append(abs(SF_t_fft[i]))

PeakFreq = np.zeros(len(Ind),dtype=float)
for i in range(0,len(Ind)):
    PeakFreq[i] = Freq[Ind[i]]
print("Peak Freq's: ",PeakFreq)
print("Amplitudes: ",Amp)

SF_fitted  = np.zeros(Nt,dtype=float)
count = 0.0
for i in range(0,Nt):
    f = Freq[i]
    for j in range(0,len(Amp)):
        SF_fitted[i] += abs(Amp[j] * (1.0/(f-PeakFreq[j]-1j*eta)).imag)
    count += SF_fitted[i] * dF

# print(count,count0)

NormCoeff1 = 1/count

NormCoeff0 = 1/count0
################ for plotting ##############

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
plt.rc('font', size=10,weight='bold')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-15,15)
ax.set_ylim(-0.1,1.2)
# major_xticks = np.arange(-15, 5, 5)
# ax.set_xticks(major_xticks)
major_yticks = np.arange(0, 1.5, 0.5)
ax.set_yticks(major_yticks)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
# ax.grid(True,which='major', color='k', linestyle='--')
ax.tick_params(direction='in',length=6, width=2)
ax.set_xlabel(r'$\omega$', fontweight='bold',fontsize=20)
ax.set_ylabel(r'$A(\omega)$', fontweight='bold',fontsize=20)
ax.tick_params(which='minor',direction='in', top=True, right=True, labelsize=16)
ax.tick_params(direction='in', width=2, top=True, right=True, labelsize=16)
# ax.set_title(r'H$_2$ CMX(K) performance, E$_{FCI}$ = -1.8510456784448648 a.u.',fontsize=10,  fontweight='bold')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

plt.fill_between(Freq,SF_f*NormCoeff0/np.pi,color='g',alpha=0.4,label='Exact')
plt.plot(Freq,SF_fitted*NormCoeff1/np.pi,'b--',linewidth=3,label=r'$\it{FFT}\{$A$^{CC}$(t)$\}$')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,fontsize=12,loc="upper right", fancybox=True, shadow=True)
plt.tight_layout()
#plt.draw()
plt.show()
