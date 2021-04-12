import numpy as np
from numpy import linalg as LA
from scipy.linalg import cosm, expm, sinm
import openfermion
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import get_sparse_operator, jordan_wigner, bravyi_kitaev
from openfermion.utils import get_ground_state, normal_ordered, commutator
from openfermion.ops import FermionOperator, SymbolicOperator
from scipy.linalg import eigh, norm
from scipy.sparse import csc_matrix

ket0  = np.array([1.,0.])
ket1  = np.array([0.,1.])

#define Pauli matrices
I2 = [[1.0,0.0],
      [0.0,1.0]]

X = [[0.0,1.0],
     [1.0,0.0]]

Y = [[0.0, complex(0,-1.0)],
     [complex(0.0,1.0),0.0]]

Z = [[1.0,0.0],
     [0.0,-1.0]]

def comm(A,B):
    return A@B - B@A

def create_AIM(num_bath_sites, U, mu, V_params, eps_params):
    '''
    Function to create a SIAM Hamiltonian for an arbitrary number of bath sites.
    Returns an object of type FermionOperator defined in openfermion.

    The states are assumed to be enumerated by all spin down occupations followed by all spin up occupations.

    The indices 0 and num_bath_sites+1 refer to the impurity site occupations (down and up respectively).
    '''
# Lists to hold the FermionOperators needed for this system and number of bath
# sites.
    c_ops = []
    cdag_ops = []

#Initialize the fermion operators that will hold the epsilon minus mu terms in
#the Hamiltonian, and the Hopping terms between the impurityand bath sites HV.
    H_eps_mu = FermionOperator()
    HV = FermionOperator()

# Loop to create the creation and annihilation operators needed.
    for i in range(0, 2*(num_bath_sites+1)):
        c_ops.append(FermionOperator( (i,0) ))
        cdag_ops.append(FermionOperator( (i,1) ))

# Loop to populate the Hopping term and potential terms.
    for i in range(0, 2*(num_bath_sites+1)):
        H_eps_mu += (eps_params[i%(num_bath_sites+1)]-mu)*cdag_ops[i]*c_ops[i]

        if i==0 or i==num_bath_sites+1:
            pass

        elif i>0 and i<num_bath_sites+1:
            HV += V_params[i-1]*(cdag_ops[0]*c_ops[i] + cdag_ops[i]*c_ops[0])

        else:
            HV += V_params[i%(num_bath_sites+2)]*(cdag_ops[num_bath_sites+1]*c_ops[i] + cdag_ops[i]*c_ops[num_bath_sites+1])
    #print(HV)
# Coulomb repulsion term on the impurity site.
    HU = U*cdag_ops[0]*c_ops[0]*cdag_ops[num_bath_sites+1]*c_ops[num_bath_sites+1]

# Combining all Hamiltonian terms
    H = H_eps_mu + HU + HV

    return H

def H_to_mat(h):
    jwh = jordan_wigner(h)
    #print(get_sparse_operator(jwh))
    #print( get_sparse_operator(jwh).toarray() )
    return get_sparse_operator(jwh).todense()

def time_evol_mat(h, total_time, num_steps, step_number):
    hmat = H_to_mat(h)
    dt = total_time/num_steps
    return expm(-1j * hmat * dt * step_number)


def create_trial_state(bitstring):
    state = np.array([1.0])
    num_els = 0
    for i in range(0, len(bitstring)):
        if bitstring[i]=='1':
            state = np.kron(state, ket1)
            num_els+=1
        else:
            state = np.kron(state, ket0)
    #print('Filling = ', num_els/len(bitstring))
    return state

def str_to_mat(string_of_paulis):
    pauli_string_mat = np.array([[1.0]])
    for i in range(0, len(string_of_paulis)):
        if string_of_paulis[i]=='X':
            pauli_string_mat = np.kron(pauli_string_mat, X)
        elif string_of_paulis[i]=='Y':
            pauli_string_mat = np.kron(pauli_string_mat, Y)
        elif string_of_paulis[i]=='Z':
            pauli_string_mat = np.kron(pauli_string_mat, Z)
        elif string_of_paulis[i]=='I':
            pauli_string_mat = np.kron(pauli_string_mat, I2)
        else:
            raise ValueError('Unknown Pauli matrix encountered!')
    return pauli_string_mat

def trot_error(num_bath_sites, u, mu, v, eps):
    term1=0.0
    term2=0.0
    eps_temp = [mu] * int(num_bath_sites+1)
    v_temp = [0] * int(num_bath_sites)
    #hu = create_AIM(num_bath_sites, u, 0.0, v_temp, eps_temp)
    h_pot = create_AIM(num_bath_sites, u, mu, v_temp, eps)
    print('H_pot: ', h_pot, '\n')
    h_hop = create_AIM(num_bath_sites, 0.0, mu, v, eps_temp)
    print('H_hop: ', h_hop, '\n')
    #print(hu,'\n', h_eps_mu,'\n', hv)
    hlist = [h_pot, h_hop]
    for i in range(0, len(hlist)):
        for j in range(i+1, len(hlist)):
            for k in range(i+1, len(hlist)):
                print(i, j, k)
                comm1 = commutator(hlist[j], hlist[i])
                #print('comm1: ', comm1, '\n')
                term1 += LA.norm(H_to_mat(commutator(hlist[k], comm1)),2)
                print('term1: ', LA.norm(H_to_mat(commutator(hlist[k], comm1)),2), '\n')
            com2 = commutator(hlist[i],hlist[j])
            #print('comm2: ', comm1, '\n')
            term2 += LA.norm(H_to_mat(commutator(hlist[i],com2)),2)
            print('term2: ', LA.norm(H_to_mat(commutator(hlist[i], com2)),2), '\n')
    vals = [term1, term2]
    return vals

#print(trot_error(2, 20.0, 10.0, [1.0,0.15], [0.0, -10.09, 20.0] ))

hpot = create_AIM(1, 8.0, 4.0, [0.0], [0.0,4.0])
hv = create_AIM(1, 0.0, 0.0, [1.0], [0.0,0.0])
#print('H: ', hv+hpot)
comm3 = H_to_mat(commutator(hv,hpot))
#print('|| [Hv, [Hv, Hu]] || : ', LA.norm(H_to_mat(commutator(hpot,commutator(hpot,hv))),2))
#print(LA.norm(comm3),2)
#trial = create_trial_state('0001111100').flatten()
#print(trial)

#A = create_AIM(4, 1.0, 1.0/2.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0])

#Ut = time_evol_mat(A, 1.0, 1.0, 1)

#pauli_string_mat = str_to_mat('YY')
#print(pauli_string_mat, '\n')

#print(trial.T@Ut@trial)

#A = create_AIM(1, 8.0, 4.0, [1.0], [0.0, 1.0])

#print(A)
