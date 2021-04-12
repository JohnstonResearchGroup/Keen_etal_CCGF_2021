import os
import sys
import math, cmath
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
#from numpy import linalg as la
#from numpy.linalg import matrix_power
from numpy import linalg as LA
from scipy.linalg import cosm, expm, sinm
from scipy.optimize import (curve_fit, minimize)
import csv
from time import perf_counter
from openfermion_funcs import *
#from qc3_spec_U8 import Ecc, H1, H2, U, mu, e0, e1, e2, V1, V2

pres = False
numup=1
numdn=2
numbath=2
numquanta=0
g=0

init_params = 'Data/init_DMFT_parameters_%dup%ddn_Nbath%03d_nquata=%03d_g=%0.3f.dat' % (numup, numdn, numbath, numquanta, g)

params = 'Data/DMFT_parameters_%dup%ddn_Nbath%03d_nquata=%03d_g=%0.3f.dat' % (numup, numdn, numbath, numquanta, g)

eps_init = np.genfromtxt(init_params, usecols=(0), dtype=float)
eps_fin = np.genfromtxt(params, usecols=(0), dtype=float)

v_init = np.genfromtxt(init_params, usecols=(1), dtype=float)
v_fin = np.genfromtxt(params, usecols=(1), dtype=float)

def norm_spec_data(df, spec_data):
    count = 0.0
    for i in range(0,len(spec_data)):
        count+= spec_data[i]*df
    return 1.0/count

U = 8
Nt   = 601
T0   = 0
T1   = 18
Time = np.linspace(T0,T1,Nt)
Ttot = T1 - T0
dT   = Ttot/(Nt-1)

Freq = np.linspace(-0.5*Nt/Ttot,0.5*Nt/Ttot,Nt)
dF   = (Nt/(Nt-1)) * (1/Ttot)

eta = 0.1

Gimpw_dat=np.genfromtxt('Data/ex_G_impw_U=%.2f_tot_time=%.2f.txt' % (U,T1), usecols=(1))

init_Gimpw_dat=np.genfromtxt('Data/init_ex_G_impw_U=%.2f_tot_time=%.2f.txt' % (U, T1), usecols=(1))

norm = norm_spec_data(dF, Gimpw_dat)
init_norm = norm_spec_data(dF, init_Gimpw_dat)

Gimpt = {}
init_Gimpt = {}
for i in range(2, 6):
    trot_steps = 2**i

    Gimpt_dat=np.genfromtxt('Data/G_impt_U=%.2f_%d_trot_steps_tot_time=%.2f.txt' % (U, trot_steps,T1), usecols=(1))

    init_Gimpt_dat=np.genfromtxt('Data/init_G_impt_U=%.2f_%d_trot_steps_tot_time=%.2f.txt' % (U, trot_steps,T1), usecols=(1))

    Gimpt.update({trot_steps: Gimpt_dat})
    init_Gimpt.update({trot_steps: init_Gimpt_dat})

ex_time_Gimpt = np.genfromtxt('Data/ex_G_impt_U=%.2f_tot_time=%.2f.txt' % (U, T1) ,usecols=(1), dtype=complex)

init_ex_time_Gimpt = np.genfromtxt('Data/init_ex_G_impt_U=%.2f_tot_time=%.2f.txt' % (U, T1) ,usecols=(1), dtype=complex)


ex_SF_t_fft = 2*np.fft.fftshift(np.fft.fft(ex_time_Gimpt))/Nt
init_ex_SF_t_fft = 2*np.fft.fftshift(np.fft.fft(init_ex_time_Gimpt))/Nt
# find peaks
ex_Amp = []
init_ex_Amp = []
ex_Ind = []
init_ex_Ind = []
for i in range(1,int(Nt/2)+1):
    tmp = abs(ex_SF_t_fft[i])
    if tmp > 0.01 and tmp > abs(ex_SF_t_fft[i+1]) and tmp > abs(ex_SF_t_fft[i-1]):
        ex_Ind.append(i)
        ex_Amp.append(abs(ex_SF_t_fft[i]))

for i in range(1,int(Nt/2)+1):
    tmp = abs(init_ex_SF_t_fft[i])
    if tmp > 0.01 and tmp > abs(init_ex_SF_t_fft[i+1]) and tmp > abs(init_ex_SF_t_fft[i-1]):
        init_ex_Ind.append(i)
        init_ex_Amp.append(abs(init_ex_SF_t_fft[i]))

ex_PeakFreq = np.zeros(len(ex_Ind),dtype=float)
for i in range(0,len(ex_Ind)):
    ex_PeakFreq[i] = Freq[ex_Ind[i]]

init_ex_PeakFreq = np.zeros(len(init_ex_Ind),dtype=float)
for i in range(0,len(init_ex_Ind)):
    init_ex_PeakFreq[i] = Freq[init_ex_Ind[i]]

ex_SF_fitted  = np.zeros(Nt,dtype=float)
for i in range(0,Nt):
    f = Freq[i]
    for j in range(0,len(ex_Amp)):
        ex_SF_fitted[i] += abs(ex_Amp[j] * (1.0/(f-ex_PeakFreq[j]-1j*eta)).imag)

init_ex_SF_fitted  = np.zeros(Nt,dtype=float)
for i in range(0,Nt):
    f = Freq[i]
    for j in range(0,len(init_ex_Amp)):
        init_ex_SF_fitted[i] += abs(init_ex_Amp[j] * (1.0/(f-init_ex_PeakFreq[j]-1j*eta)).imag)

fitted_norm = norm_spec_data(dF, ex_SF_fitted.imag)
init_fitted_norm = norm_spec_data(dF, init_ex_SF_fitted.imag)
#print("Initial Peak Freq's for", trot_steps, 'trotter steps: ',ex_init_PeakFreq[trot_steps])
#print("Initial Amplitudes for", trot_steps, 'trotter steps: ', init_Amps[trot_steps], '\n')




SF_t_fft = {}
init_SF_t_fft = {}
SF_fitted = {}
init_SF_fitted ={}
Amps = {}
Inds = {}
PeakFreq = {}
init_Amps = {}
init_Inds = {}
init_PeakFreq = {}
norm_coeff = {}
init_norm_coeff = {}
for k in range(2,6):
    trot_steps = 2**k
    FFT_dat = 2*np.fft.fftshift(np.fft.fft(Gimpt[trot_steps]))/Nt
    init_FFT_dat = 2*np.fft.fftshift(np.fft.fft(init_Gimpt[trot_steps]))/Nt

    SF_t_fft.update({trot_steps: FFT_dat})
    init_SF_t_fft.update({trot_steps: init_FFT_dat})
    # find peaks
    Amps.update({trot_steps: []})
    Inds.update({trot_steps: []})
    init_Amps.update({trot_steps: []})
    init_Inds.update({trot_steps: []})
    for i in range(1,int(Nt/2)+1):
        tmp = abs(SF_t_fft[trot_steps][i])
        init_tmp = abs(init_SF_t_fft[trot_steps][i])
        if tmp > 0.01 and tmp > abs(SF_t_fft[trot_steps][i+1]) and tmp > abs(SF_t_fft[trot_steps][i-1]):
            Inds[trot_steps].append(i)
            Amps[trot_steps].append(abs(SF_t_fft[trot_steps][i]))
        else:
            pass
        if init_tmp > 0.01 and init_tmp > abs(init_SF_t_fft[trot_steps][i+1]) and init_tmp > abs(init_SF_t_fft[trot_steps][i-1]):
            init_Inds[trot_steps].append(i)
            init_Amps[trot_steps].append(abs(init_SF_t_fft[trot_steps][i]))
        else:
            pass

    PeakFreq.update({trot_steps: np.zeros(len(Inds[trot_steps]),dtype=float)})
    init_PeakFreq.update({trot_steps: np.zeros(len(init_Inds[trot_steps]),dtype=float)})
    if len(Inds[trot_steps])==0:
        pass
    else:
        for i in range(0,len(Inds[trot_steps])):
            PeakFreq[trot_steps][i] = Freq[Inds[trot_steps][i]]

    if len(init_Inds[trot_steps])==0:
        pass
    else:
        for i in range(0,len(init_Inds[trot_steps])):
            init_PeakFreq[trot_steps][i] = Freq[init_Inds[trot_steps][i]]

    SF_fitted.update({trot_steps: np.zeros(Nt,dtype=float)})
    init_SF_fitted.update({trot_steps: np.zeros(Nt,dtype=float)})
    count = 0.0
    init_count = 0.0
    for i in range(0, Nt):
        f1 = Freq[i]
        for j in range(0,len(Amps[trot_steps])):
            SF_fitted[trot_steps][i] += Amps[trot_steps][j] * (1.0/(f1-PeakFreq[trot_steps][j]-1j*eta)).imag
        NormCoeff = norm_spec_data(dF, SF_fitted[trot_steps])

    norm_coeff.update({trot_steps: NormCoeff})
    print("Peak Freq's for", trot_steps, 'trotter steps: ',PeakFreq[trot_steps])
    print("Amplitudes for", trot_steps, 'trotter steps: ', Amps[trot_steps], '\n')

    for i in range(0, Nt):
        f1 = Freq[i]
        for j in range(0,len(init_Amps[trot_steps])):
            init_SF_fitted[trot_steps][i] += init_Amps[trot_steps][j] * (1.0/(f1-init_PeakFreq[trot_steps][j]-1j*eta)).imag
            '''
            init_count += (1.0/np.pi)*init_SF_fitted[trot_steps][i] * dF
        init_NormCoeff = 1.0/init_count
            '''
        init_NormCoeff = norm_spec_data(dF, init_SF_fitted[trot_steps])
    init_norm_coeff.update({trot_steps: init_NormCoeff})
    print("Initial Peak Freq's for", trot_steps, 'trotter steps: ',init_PeakFreq[trot_steps])
    print("Initial Amplitudes for", trot_steps, 'trotter steps: ', init_Amps[trot_steps], '\n')

print('norm coeffs: ', norm_coeff)
print('init norm coeffs: ', init_norm_coeff)

#c2 = 0
#for i in range(0,Nt):
#    c2+=init_SF_fitted[4][i]*init_norm_coeff[4]*dF
#print(c2)
#mpl.rcParams['axes.linewidth'] = 2.0
#mpl.rcParams['xtick.major.width'] = 2
#mpl.rcParams['ytick.major.width'] = 2
#plt.rcParams["font.family"] = "Arial"
#plt.rc('font', size=11,weight='normal')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=11,weight='normal')

if pres==True:
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 4.5), dpi=80,sharex=True, sharey=True)
    default_fontsize=19
    leg_fontsize=default_fontsize-4

    axs[0].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal', fontsize = default_fontsize)
    axs[1].set_xlabel(r'$\omega', fontweight='normal',fontsize=default_fontsize)
    axs[1].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal',fontsize = default_fontsize)

else:
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.5), dpi=80,sharex=True, sharey=True)
    default_fontsize=11
    leg_fontsize = default_fontsize

    axs[0].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal', fontsize = default_fontsize)
    axs[1].set_xlabel(r'$\omega$', fontweight='normal',fontsize=default_fontsize)
    axs[1].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal',fontsize = default_fontsize)


#plt.ion()
#plt.rc('axes', titlesize=12)
#plt.rcParams.update({'font.size': 22})
axs[0].ticklabel_format(style='plain')
axs[1].ticklabel_format(style='plain')
axs[0].set_xlim(-8, 0)
axs[0].set_xlim(-8, 0)
#axs[0].set_ylim(0.0,1.0)
#axs[0].set_yticks(np.arange(0,1.25,0.25))
#axs[0].set_yticklabels(np.arange(0.0,1.25,0.25))
axs[0].set_xticks(np.arange(-8,1))
#axs[0].set_xticklabels([-6,'' , -4, '', -2,'' , 0,'' , 2,'' , 4,'' , 6], fontweight='normal', fontsize=11)
axs[0].set_xticklabels([-8 ,'' ,-6,'' , -4, '', -2,'' , 0], fontweight='normal', fontsize=default_fontsize)
#axs[1].set_yticks(np.arange(0,2,0.5))
#axs[1].set_yticklabels(np.arange(0,2,0.5))


#axs[1].plot(Freq.real,ex_SF_fitted*fitted_norm, 'g--', label='Exact')
#axs[0].plot(Freq,init_ex_SF_fitted*init_fitted_norm, 'g--',label='Exact')

for k in range(2,6):
    trot_steps = 2**k

    axs[1].plot(Freq.real,SF_fitted[trot_steps]*norm_coeff[trot_steps],'o', markersize=2.0, color='orange', label= str(trot_steps)+' steps')

    axs[0].plot(Freq, init_SF_fitted[trot_steps]*init_norm_coeff[trot_steps], 'o', markersize=2.0, color='orange', label=str(trot_steps)+' steps')
    #axs[1].plot(Freq.real,SF_t_fft[trot_steps].imag*norm_coeff[trot_steps],'o', markersize=2.0, label= str(trot_steps)+' steps')

    #axs[0].plot(Freq, init_SF_t_fft[trot_steps].imag*init_norm_coeff[trot_steps], 'o', markersize=2.0, label=str(trot_steps)+' steps')
axs[1].plot(Freq.real,Gimpw_dat*norm, 'b--', label='Exact')
axs[0].plot(Freq,init_Gimpw_dat*init_norm, 'b--', label='Exact')
#axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_popt), 'r--')
#axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_exact_popt),'g-')

#axs[0].set(ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
#axs[1].set(xlabel=r't$[1/t*]$',ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)

#axs[0].legend(loc='upper right' ,frameon=False, fontsize=leg_fontsize)

axs[0].annotate("c", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
axs[1].annotate("d", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
'''
axs[1].annotate("eps1 = " + str(round(eps_fin[0], 3))+",  v1 = " + str(round(v_fin[0],3)), xy=(0.72, 0.9), xycoords="axes fraction", fontweight='bold')
axs[1].annotate("eps2 = " + str(round(eps_fin[1],3))+",  v2 = " + str(round(v_fin[1],3)), xy=(0.72, 0.8), xycoords="axes fraction", fontweight='bold')

axs[0].annotate("eps1 = " + str(round(eps_init[0],3))+",  v1 = " + str(round(v_init[0],3)), xy=(0.72, 0.9), xycoords="axes fraction", fontweight='bold')
axs[0].annotate("eps2 = " + str(round(eps_init[1],3))+",  v2 = " + str(round(v_init[1],3)), xy=(0.72, 0.8), xycoords="axes fraction", fontweight='bold')
'''
plt.tight_layout()
#plt.savefig(sys.argv[2])
#plt.draw()

if pres==True:
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=80,sharex=True, sharey=True)
    default_fontsize=19
    leg_fontsize=default_fontsize-5

    axs[0].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal', fontsize = default_fontsize)
    axs[1].set_xlabel(r't [1/t*]', fontweight='normal',fontsize=default_fontsize)
    axs[0].set_xlabel(r't [1/t*]', fontweight='normal',fontsize = default_fontsize)
else:
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.5), dpi=80,sharex=True, sharey=True)
    default_fontsize=11
    leg_fontsize = default_fontsize

    axs[0].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal', fontsize = default_fontsize)
    axs[1].set_xlabel(r't', fontweight='normal',fontsize=default_fontsize)
    axs[1].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal',fontsize = default_fontsize)



#plt.ion()
#plt.rc('axes', titlesize=12)
#plt.rcParams.update({'font.size': 22})
axs[0].ticklabel_format(style='plain')
axs[1].ticklabel_format(style='plain')
axs[0].set_xlim(0, 2)
axs[1].set_xlim(0, 2)
axs[0].set_xticks(np.arange(0,2.25,0.25))
axs[0].set_xticklabels([0,'' , 0.5, '', 1.0,'' , 1.5,'' , 2], fontweight='normal', fontsize=default_fontsize)
axs[1].set_xticklabels([0,'' , 0.5, '', 1.0,'' , 1.5,'' , 2], fontweight='normal', fontsize=default_fontsize)
axs[1].set_yticks(np.arange(-1,1.25,0.25))
axs[0].set_yticklabels([-1,'' , -0.5, '', 0.0,'' , 0.5,'' , 1.0], fontweight='normal', fontsize=default_fontsize)


for k in [3,5]:
    trot_steps = 2**k

    axs[0].plot(Time,init_Gimpt[trot_steps],'o', color='orange', markersize=4.0,label= str(trot_steps)+' steps')

    axs[1].plot(Time,Gimpt[trot_steps],'o', color='orange', markersize=4.0,label=str(trot_steps)+' steps')
axs[0].plot(Time,init_ex_time_Gimpt, 'b--', label='Exact')
axs[1].plot(Time,ex_time_Gimpt, 'b--',label='Exact')

#axs[0].legend(loc='upper right' ,ncol=2, frameon=False, fontsize=leg_fontsize)

#axs[0].set(ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
#axs[1].set(xlabel=r't$[1/t*]$',ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)

axs[0].annotate("c", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
axs[1].annotate("d", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')

plt.tight_layout()


plt.show()
