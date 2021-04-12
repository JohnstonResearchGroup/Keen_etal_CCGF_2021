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
from scipy.linalg import cosm, expm, sinm
from scipy.optimize import (curve_fit, minimize)
from Cluster_ops import *
from Measurement_Circuits import *
from Pauli_matrices import *
from Params import *
from indexed import IndexedOrderedDict
import csv
from time import perf_counter

pres = False

V1 = float(sys.argv[1])
V2 = float(sys.argv[2])

_label1 = "V = "+str(V1)
_label2 = "V = "+str(V2)



def E0N(hamiltonian, trial_vec, u, v):
    return (np.conjugate(trial_vec).transpose()@expm(-T_op(u,v))@hamiltonian@expm(T_op(u,v))@trial_vec)



QC_data_file1 = 'Time_Domain_Data/Quantum/new_QC_G_impt_V=%.2f_%d_steps_tot_time3=%.1f_error_rates=(%.2e,%.2e).csv' % (V1, steps,total_time, prob_1, prob_2)

QC_data_file2 = 'Time_Domain_Data/Quantum/new_QC_G_impt_V=%.2f_%d_steps_tot_time3=%.1f_error_rates=(%.2e,%.2e).csv' % (V2, steps,total_time, prob_1, prob_2)

classical_trot_data_file1 = 'Time_Domain_Data/Classical/newCC_G_impt_trot_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V1, steps,total_time)

classical_trot_data_file2 = 'Time_Domain_Data/Classical/newCC_G_impt_trot_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V2, steps,total_time)

classical_data_file1 = 'Time_Domain_Data/Classical/newCC_G_impt_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V1, steps2,total_time2)

classical_data_file2 = 'Time_Domain_Data/Classical/newCC_G_impt_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V2, steps2,total_time2)

ex_time_Gimpt1 = np.genfromtxt('Time_Domain_Data/Classical/newCC_G_impt_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V1, 50000,50), delimiter='\t', usecols=(1), dtype=complex)

ex_time_Gimpt2 = np.genfromtxt('Time_Domain_Data/Classical/newCC_G_impt_V=%.2f_%d_steps_tot_time=%.1f.csv' % (V2, 50000,50), delimiter='\t', usecols=(1), dtype=complex)

classical_Gimpt_trot1 = np.genfromtxt(classical_trot_data_file1, delimiter='\t', usecols=(1), dtype=complex)

classical_Gimpt_exact1 = np.genfromtxt(classical_data_file1, delimiter='\t', usecols=(1), dtype=complex)

classical_Gimpt_trot2 = np.genfromtxt(classical_trot_data_file2, delimiter='\t', usecols=(1), dtype=complex)

classical_Gimpt_exact2 = np.genfromtxt(classical_data_file2, delimiter='\t', usecols=(1), dtype=complex)

#all_term_QC_Gimpt = np.genfromtxt(QC_data_file, delimiter='\t', usecols=(1), dtype=complex)

#three_term_QC_Gimpt = np.genfromtxt(QC_data_file, delimiter='\t', usecols=(2), dtype=complex)

Ur_only_QC_Gimpt1 = np.genfromtxt(QC_data_file1, delimiter='\t', usecols=(1), dtype=complex)

three_term_QC_Gimpt_Ur_only1 = np.genfromtxt(QC_data_file1, delimiter='\t', usecols=(2), dtype=complex)

Ur_only_QC_Gimpt2 = np.genfromtxt(QC_data_file2, delimiter='\t', usecols=(1), dtype=complex)

three_term_QC_Gimpt_Ur_only2 = np.genfromtxt(QC_data_file2, delimiter='\t', usecols=(2), dtype=complex)

#f=plt.figure(1)
#plt.plot(time_arr, all_term_QC_Gimpt.imag, label='QC All')
#plt.plot(time_arr, three_term_QC_Gimpt.imag, label='QC Three terms')
#plt.legend()

SF_t_fft1 = 2*np.fft.fftshift(np.fft.fft(three_term_QC_Gimpt_Ur_only1))/steps
ex_SF_t_fft1 = 2*np.fft.fftshift(np.fft.fft(classical_Gimpt_exact1))/steps2

SF_t_fft2 = 2*np.fft.fftshift(np.fft.fft(three_term_QC_Gimpt_Ur_only2))/steps
ex_SF_t_fft2 = 2*np.fft.fftshift(np.fft.fft(classical_Gimpt_exact2))/steps2


# find peaks
Amp1 = []
Ind1 = []
for i in range(1,int(steps/2)+1):
    tmp = abs(SF_t_fft1[i])
    if tmp > 0.1 and tmp > abs(SF_t_fft1[i+1]) and tmp > abs(SF_t_fft1[i-1]):
        Ind1.append(i)
        Amp1.append(abs(SF_t_fft1[i]))

PeakFreq1 = np.zeros(len(Ind1),dtype=float)
if len(Ind1)==0:
    pass
else:
    for i in range(0,len(Ind1)):
        PeakFreq1[i] = freq_arr[Ind1[i]]

SF_fitted1  = np.zeros(steps+1,dtype=complex)
count1 = 0.0
if not Amp1:
    NormCoeff1=1.0
else:
    for i in range(0, steps+1):
        f1 = freq_arr[i]
        SF_fitted1[i] = Amp1[0] * (1.0/(f1-PeakFreq1[0]-1j*eta) - 1.0/(f1+PeakFreq1[0]+1j*eta))
        for j in range(1,len(Amp1)):
            SF_fitted1[i] += Amp1[j] * (1.0/(f1-PeakFreq1[j]-1j*eta) - 1.0/(f1+PeakFreq1[j]+1j*eta))
        count1 += SF_fitted1[i] * dw
    NormCoeff1 = 1.0/count1
print("Peak Freq's: ",PeakFreq1)
print("Amplitudes: ", Amp1)

# find peaks
Amp2 = []
Ind2 = []
for i in range(1,int(steps/2)+1):
    tmp = abs(SF_t_fft2[i])
    if tmp > 0.1 and tmp > abs(SF_t_fft2[i+1]) and tmp > abs(SF_t_fft2[i-1]):
        Ind2.append(i)
        Amp2.append(abs(SF_t_fft2[i]))

PeakFreq2 = np.zeros(len(Ind2),dtype=float)
if len(Ind2)==0:
    pass
else:
    for i in range(0,len(Ind2)):
        PeakFreq2[i] = freq_arr[Ind2[i]]

SF_fitted2  = np.zeros(steps+1,dtype=complex)
count2 = 0.0
if not Amp2:
    NormCoeff2=1.0
else:
    for i in range(0, steps):
        f2 = freq_arr[i]
        SF_fitted2[i] = Amp2[0] * (1.0/(f2-PeakFreq2[0]-1j*eta) - 1.0/(f2+PeakFreq2[0]+1j*eta))
        for j in range(1,len(Amp2)):
            SF_fitted2[i] += Amp2[j] * (1.0/(f2-PeakFreq2[j]-1j*eta) - 1.0/(f2+PeakFreq2[j]+1j*eta))
        count2 += SF_fitted2[i] * dw
    NormCoeff2 = 1.0/count2
print("Peak Freq's: ",PeakFreq2)
print("Amplitudes: ", Amp2)


# find peaks
ex_Amp1 = []
ex_Ind1 = []
for i in range(1,int(steps2/2)+1):
    tmp = abs(ex_SF_t_fft1[i])
    if tmp > 0.1 and tmp > abs(ex_SF_t_fft1[i+1]) and tmp > abs(ex_SF_t_fft1[i-1]):
        ex_Ind1.append(i)
        ex_Amp1.append(abs(ex_SF_t_fft1[i]))
        #print(ex_Amp1)

ex_PeakFreq1 = np.zeros(len(ex_Ind1),dtype=float)
if len(ex_Ind1)==0:
    pass
else:
    for i in range(0,len(ex_Ind1)):
        ex_PeakFreq1[i] = fine_freq_arr[ex_Ind1[i]]

ex_SF_fitted1  = np.zeros(steps2+1,dtype=complex)
ex_count1 = 0.0
if not ex_Amp1:
    ex_NormCoeff1=1.0
else:
    for i in range(0, steps2):
        f1 = fine_freq_arr[i]
        ex_SF_fitted1[i] = ex_Amp1[0] * (1.0/(f1-ex_PeakFreq1[0]-1j*eta) - 1.0/(f1+ex_PeakFreq1[0]+1j*eta))
        for j in range(1,len(ex_Amp1)):
            ex_SF_fitted1[i] += ex_Amp1[j] * (1.0/(f1-ex_PeakFreq1[j]-1j*eta) - 1.0/(f1+ex_PeakFreq1[j]+1j*eta))
        ex_count1 += ex_SF_fitted1[i] * fine_dw
    ex_NormCoeff1 = 1.0/ex_count1
print("Exact Peak Freq's: ",ex_PeakFreq1)
print("Exact Amplitudes: ", ex_Amp1)

# find peaks
ex_Amp2 = []
ex_Ind2 = []
for i in range(1,int(steps2/2)+1):
    tmp = abs(ex_SF_t_fft2[i])
    if tmp > 0.1 and tmp > abs(ex_SF_t_fft2[i+1]) and tmp > abs(ex_SF_t_fft2[i-1]):
        ex_Ind2.append(i)
        ex_Amp2.append(abs(ex_SF_t_fft2[i]))
        #print(ex_Amp2)

ex_PeakFreq2 = np.zeros(len(ex_Ind2),dtype=float)
if len(ex_Ind2)==0:
    pass
else:
    for i in range(0,len(ex_Ind2)):
        ex_PeakFreq2[i] = fine_freq_arr[ex_Ind2[i]]

ex_SF_fitted2  = np.zeros(steps2+1,dtype=complex)
ex_count2 = 0.0
if not ex_Amp2:
    ex_NormCoeff2=1.0
else:
    for i in range(0, steps2):
        f2 = fine_freq_arr[i]
        ex_SF_fitted2[i] = ex_Amp2[0] * (1.0/(f2-ex_PeakFreq2[0]-1j*eta) - 1.0/(f2+ex_PeakFreq2[0]+1j*eta))
        for j in range(1,len(ex_Amp2)):
            ex_SF_fitted2[i] += ex_Amp2[j] * (1.0/(f2-ex_PeakFreq2[j]-1j*eta) - 1.0/(f2+ex_PeakFreq2[j]+1j*eta))
        ex_count2 += ex_SF_fitted2[i] * fine_dw
    ex_NormCoeff2 = 1.0/ex_count2
print("Exact Peak Freq's: ",ex_PeakFreq2)
print("Exact Amplitudes: ", ex_Amp2)

#mpl.rcParams['axes.linewidth'] = 2.0
#mpl.rcParams['xtick.major.width'] = 2
#mpl.rcParams['ytick.major.width'] = 2
#plt.rcParams["font.family"] = "Arial"
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=11,weight='normal')
#plt.rc('font', size=11,weight='normal')
if pres==True:
    plt.rc('font', size=19,weight='normal')
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 4.5), dpi=80, sharey=True, sharex=True)
    #plt.ion()
    #plt.rc('axes', titlesize=12)
    #plt.rcParams.update({'font.size': 22})
    axs[0].ticklabel_format(style='plain')
    axs[1].ticklabel_format(style='plain')
    axs[0].set_xlim(-6, 6)
    axs[0].set_xlim(-6, 6)
    axs[0].set_xticks(np.arange(-6,7))
    axs[0].set_xticklabels([-6,'' , -4, '', -2,'' , 0,'' , 2,'' , 4,'' , 6], fontweight='normal', fontsize=19)
    axs[1].set_yticks(np.arange(0,2,0.5))
    axs[1].set_yticklabels(np.arange(0,2,0.5))

    axs[0].plot(freq_arr, SF_fitted1*NormCoeff1 ,'o', label='Simulator')
    axs[0].plot(freq_arr,ex_SF_fitted1*ex_NormCoeff1, 'b--', label='Exact')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *popt), 'r--', label='Fit')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *exact_popt), 'g-', label='Exact')
    #axs[0].plot(time_arr, -1.0*trotter.imag,'v', markersize=8.0, label='Trotter')

    axs[1].plot(freq_arr, SF_fitted2*NormCoeff2, 'o' ,label='Simulator')
    axs[1].plot(freq_arr,ex_SF_fitted2*ex_NormCoeff2, 'b--',label='Exact')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_popt), 'r--')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_exact_popt),'g-')

    #axs[0].set(ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    #axs[1].set(xlabel=r't$[1/t*]$',ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    axs[0].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal', fontsize = 19)
    axs[1].set_xlabel(r'$\omega$', fontweight='normal',fontsize=19)
    axs[0].set_xlabel(r'$\omega$', fontweight='normal',fontsize=19)
    #axs[1].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal',fontsize = 11)
    axs[0].legend(loc='upper center', frameon=False, fontsize=15)

    axs[0].annotate("a", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
    axs[1].annotate("b", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')

    plt.tight_layout()
    #plt.savefig(sys.argv[2])
    #plt.draw()

else:
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.5), dpi=80,sharex=True, sharey=True)
    #plt.ion()
    #plt.rc('axes', titlesize=12)
    #plt.rcParams.update({'font.size': 22})
    axs[0].ticklabel_format(style='plain')
    axs[1].ticklabel_format(style='plain')
    axs[0].set_xlim(-6, 6)
    axs[0].set_xlim(-6, 6)
    axs[0].set_xticks(np.arange(-6,7))
    axs[0].set_xticklabels([-6,'' , -4, '', -2,'' , 0,'' , 2,'' , 4,'' , 6], fontweight='normal', fontsize=11)
    axs[1].set_yticks(np.arange(0,2,0.5))
    axs[1].set_yticklabels(np.arange(0,2,0.5))

    axs[0].plot(freq_arr, SF_fitted1*NormCoeff1 ,'o', color='orange',markersize=2.0 , label='Simulator')
    axs[0].plot(freq_arr,ex_SF_fitted1*ex_NormCoeff1, 'b--', label='Exact')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *popt), 'r--', label='Fit')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *exact_popt), 'g-', label='Exact')
    #axs[0].plot(time_arr, -1.0*trotter.imag,'v', markersize=8.0, label='Trotter')

    axs[1].plot(freq_arr, SF_fitted2*NormCoeff2, 'o', color='orange', markersize=2.0 ,label='Simulator')
    axs[1].plot(freq_arr,ex_SF_fitted2*ex_NormCoeff2, 'b--',label='Exact')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_popt), 'r--')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_exact_popt),'g-')

    #axs[0].set(ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    #axs[1].set(xlabel=r't$[1/t*]$',ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    axs[0].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal', fontsize = 11)
    axs[1].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal', fontsize = 11)
    axs[1].set_xlabel(r'$\omega$', fontweight='normal',fontsize=11)
    #axs[0].set_xlabel(r'$\omega$', fontweight='normal',fontsize=11)
    #axs[1].set_ylabel(r'$A(\omega + $i$\delta)$', fontweight='normal',fontsize = 11)
    #axs[0].legend(loc='upper center', frameon=False, fontsize=10)

    axs[0].annotate("a", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
    axs[1].annotate("b", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')

    plt.tight_layout()
    #plt.savefig(sys.argv[2])
    #plt.draw()


#mpl.rcParams['axes.linewidth'] = 2.0
#mpl.rcParams['xtick.major.width'] = 2
#mpl.rcParams['ytick.major.width'] = 2
plt.rcParams["font.family"] = "Arial"
#plt.rcParams["font"] = "Arial"
plt.rc('font', size=11,weight='normal')
if pres==True:
    plt.rc('font', size=19,weight='normal')
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=80, sharex=True, sharey=True)
    #plt.ion()
    #plt.rc('axes', titlesize=12)
    #plt.rcParams.update({'font.size': 22})
    axs[0].ticklabel_format(style='plain')
    axs[1].ticklabel_format(style='plain')
    axs[0].set_xlim(0, 2)
    axs[1].set_xlim(0, 2)
    axs[0].set_xticks(np.arange(0,2.25,0.25))
    axs[0].set_xticklabels([0,'' , 0.5, '', 1.0,'' , 1.5,'' , 2], fontweight='normal', fontsize=19)
    axs[1].set_yticks(np.arange(-1,1.25,0.25))
    axs[1].set_yticklabels([-1,'' , -0.5, '', 0.0,'' , 0.5,'' , 1.0], fontweight='normal', fontsize=19)

    axs[0].plot(time_arr, 2*three_term_QC_Gimpt_Ur_only1.imag ,'o',color='orange', markersize=4.0,label='Simulator')
    axs[0].plot(fine_time_arr,2*ex_time_Gimpt1.imag,'b--', label='Exact')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *popt), 'r--', label='Fit')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *exact_popt), 'g-', label='Exact')
    #axs[0].plot(time_arr, -1.0*trotter.imag,'v', markersize=8.0, label='Trotter')

    axs[1].plot(time_arr, 2*three_term_QC_Gimpt_Ur_only2.imag ,'o', color='orange', markersize=4.0,label='Simulator')
    axs[1].plot(fine_time_arr,2*ex_time_Gimpt2.imag, 'b--', label='Exact')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_popt), 'r--')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_exact_popt),'g-')

    #axs[0].set(ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    #axs[1].set(xlabel=r't$[1/t*]$',ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    axs[0].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal', fontsize = 19)
    axs[0].set_xlabel(r't [1/t*]', fontweight='normal',fontsize=19)
    axs[1].set_xlabel(r't [1/t*]', fontweight='normal',fontsize=19)
    #axs[1].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal',fontsize = 11)
    axs[0].legend(frameon=False, fontsize=13, loc='upper left')
else:
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.5), dpi=80, sharex=True, sharey=True)
    #plt.ion()
    #plt.rc('axes', titlesize=12)
    #plt.rcParams.update({'font.size': 22})
    axs[0].ticklabel_format(style='plain')
    axs[1].ticklabel_format(style='plain')
    axs[0].set_xlim(0, 2)
    axs[1].set_xlim(0, 2)
    axs[0].set_xticks(np.arange(0,2.25,0.25))
    axs[0].set_xticklabels([0,'' , 0.5, '', 1.0,'' , 1.5,'' , 2], fontweight='normal', fontsize=11)
    axs[1].set_yticks(np.arange(-1,1.25,0.25))
    axs[1].set_yticklabels([-1,'' , -0.5, '', 0.0,'' , 0.5,'' , 1.0], fontweight='normal', fontsize=11)

    axs[0].plot(time_arr, 2*three_term_QC_Gimpt_Ur_only1.imag ,'o',color='orange', markersize=4.0,label='Simulator')
    axs[0].plot(fine_time_arr,2*ex_time_Gimpt1.imag, 'b--', label='Exact')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *popt), 'r--', label='Fit')
    #axs[0].plot(fine_time_arr, fit(fine_time_arr, *exact_popt), 'g-', label='Exact')
    #axs[0].plot(time_arr, -1.0*trotter.imag,'v', markersize=8.0, label='Trotter')

    axs[1].plot(time_arr, 2*three_term_QC_Gimpt_Ur_only2.imag ,'o',color='orange', markersize=4.0,label='Simulator')
    axs[1].plot(fine_time_arr,2*ex_time_Gimpt2.imag, 'b--' , label='Exact')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_popt), 'r--')
    #axs[1].plot(fine_time_arr, fit(fine_time_arr, *fin_exact_popt),'g-')

    #axs[0].set(ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    #axs[1].set(xlabel=r't$[1/t*]$',ylabel=r'-Im[$G_{imp}^R(t)$]', fontsize=22)
    axs[0].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal', fontsize = 11)
    axs[1].set_xlabel(r't', fontweight='normal',fontsize=11)
    axs[1].set_ylabel(r'$\mathregular{Im[G_{imp}(t)]}$', fontweight='normal',fontsize = 11)
    #axs[0].legend(frameon=False, fontsize=10)

    axs[0].annotate("a", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')
    axs[1].annotate("b", xy=(0.02, 0.9), xycoords="axes fraction", fontweight='bold')

plt.tight_layout()

plt.show()
