import os
import sys
import numpy as np
from Pauli_matrices import *
import qiskit.providers.aer.noise as noise
from Cluster_ops import *
V = float(sys.argv[1])

bit_flip=False

if bit_flip==True:
    # Error probabilities
    prob_1 = 1.0e-2  # 1-qubit gate
    prob_2 = 1.0e-1  # 2-qubit gate

    bit_flip = noise.pauli_error([('X', prob_1), ('I', 1 - prob_1)])
    cnot_flip = noise.pauli_error([('I', 1)]).tensor(noise.pauli_error([('X', prob_1), ('I', 1 - prob_1)]))

    # Depolarizing quantum errors
    #error_1 = noise.depolarizing_error(prob_1, 1)
    #error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(cnot_flip, ['cx'])

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates
else:
    # Error probabilities
    #prob_1 = 1.0e-4  # 1-qubit gate
    #prob_2 = 1.0e-3  # 2-qubit gate
    prob_1=0.0
    prob_2=0.0
    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

#Do you want to draw the circuit?
draw_circ=True

#First or second order product formula?
second_order = True

twopi = 2.0*np.pi

steps = 500
steps2 = 500
total_time = 10.0
total_time2 = 10.0
dt = total_time/steps
dt2 = total_time2/steps2
time_arr = np.linspace(0.0, total_time, steps+1, dtype=float)
time_arr2 = np.linspace(0.0, total_time2, steps2+1, dtype=float)
fine_time_arr = np.linspace(0.0, 50, 50001, dtype=float)

num_trot = 1

freq_arr = np.linspace(-0.5*steps/total_time, 0.5*steps/total_time, steps+1)
dw   = (steps/(steps-1)) * (1/total_time)

fine_freq_arr = np.linspace(-0.5*steps2/total_time2, 0.5*steps2/total_time2, steps2)
fine_dw   = (steps2/(steps2-1)) * (1/total_time2)

U=8.0
mu = U/2.0
epc = 0.0
#V=1.0e-2
eta=1.0e-1

'''
scalars for "a_p exp(T) |0>" and "a_p^\dagger exp(T) |0>"
'''
mu1 = 1.0
mu2 = t21_val(U,V)

'''
scalars for "<0| (1+\Lambda) exp(-T) a_p^\dagger"
and "<0| (1+\Lambda) exp(-T) a_p"
'''
nu1 = 1.0 - lam12_val(U,V)*t21_val(U,V) - lamdoub_val(U,V)*(tdoub_val(U,V) +t21_val(U,V)**2)
nu2 = lam12_val(U,V) - lamdoub_val(U,V)*t21_val(U,V)

H_SIAM =  U/4.0 * (Z1Z3 - Z1 - Z3 + I) + mu/2.0 * (Z1 + Z3) \
         +V/2.0 * (X1X2 + Y1Y2 + X3X4 + Y3Y4) - U/2.0 * I

ket0 = [1.0, 0.0]
ket1 = [0.0, 1.0]
trial_state = np.kron(np.kron(ket0, ket1),np.kron(ket1, ket0))
trial_state_dag = np.conj(trial_state).transpose()
shot=10000
