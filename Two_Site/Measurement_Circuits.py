import math, cmath
from Cluster_ops import *
from Params import *
#Remember to enter qubits as register[qubit num]

def trial_state_prep(circ, qubit1, qubit2, qubit3, qubit4):
    circ.x(qubit2)
    circ.x(qubit3)
    return circ

def first_ord_exp_sigma_single_ex_new(u, v, circ, ancilla1, qubit1, qubit2, qubit3, qubit4):
    circ.cu1(-np.pi/2.0, ancilla1, qubit1) #u1(-pi/2) = S^\dagger
    circ.ccx(ancilla1, qubit1, qubit2)
    circ.crx(-lam12_val(u,v).real, ancilla1, qubit1)
    circ.ch(ancilla1, qubit1)
    circ.ccx(ancilla1, qubit1, qubit2)
    circ.cu1(np.pi/2.0, ancilla1, qubit1)
    circ.ch(ancilla1, qubit1)
    circ.crz(lam12_val(u,v).real, ancilla1, qubit2)
    circ.ccx(ancilla1, qubit1, qubit2)
    circ.crx(-np.pi/2.0, ancilla1, qubit1)
    circ.crx(np.pi/2.0, ancilla1, qubit2)
    circ.cu1(np.pi/2.0, ancilla1, qubit1) #u1(pi/2) = S

    circ.cu1(-np.pi/2.0, ancilla1, qubit3)
    circ.ccx(ancilla1, qubit3, qubit4)
    circ.crx(-lam12_val(u,v).real, ancilla1, qubit3)
    circ.ch(ancilla1, qubit3)
    circ.ccx(ancilla1, qubit3, qubit4)
    circ.cu1(np.pi/2.0, ancilla1, qubit3)
    circ.ch(ancilla1, qubit3)
    circ.crz(lam12_val(u,v).real, ancilla1, qubit4)
    circ.ccx(ancilla1, qubit3, qubit4)
    circ.crx(-np.pi/2.0, ancilla1, qubit3)
    circ.crx(np.pi/2.0, ancilla1, qubit4)
    circ.cu1(np.pi/2.0, ancilla1, qubit3)

    return circ

def sec_ord_exp_sigma_single_ex_new(u, v, circ, ancilla1, qubit1, qubit2, qubit3, qubit4):
    circ.cu1(-np.pi/2.0, ancilla1, qubit1) #u1(-pi/2) = S^\dagger
    circ.ccx(ancilla1, qubit1, qubit2)
    circ.crx(-lam12_val(u,v).real/2.0, ancilla1, qubit1)
    circ.ch(ancilla1, qubit1)
    circ.ccx(ancilla1, qubit1, qubit2)
    circ.cu1(np.pi/2.0, ancilla1, qubit1)
    circ.ch(ancilla1, qubit1)
    circ.crz(lam12_val(u,v).real/2.0, ancilla1, qubit2)
    circ.ccx(ancilla1, qubit1, qubit2)
    circ.crx(-np.pi/2.0, ancilla1, qubit1)
    circ.crx(np.pi/2.0, ancilla1, qubit2)
    circ.cu1(np.pi/2.0, ancilla1, qubit1) #u1(pi/2) = S

    circ.cu1(-np.pi/2.0, ancilla1, qubit3)
    circ.ccx(ancilla1, qubit3, qubit4)
    circ.crx(-lam12_val(u,v).real/2.0, ancilla1, qubit3)
    circ.ch(ancilla1, qubit3)
    circ.ccx(ancilla1, qubit3, qubit4)
    circ.cu1(np.pi/2.0, ancilla1, qubit3)
    circ.ch(ancilla1, qubit3)
    circ.crz(lam12_val(u,v).real/2.0, ancilla1, qubit4)
    circ.ccx(ancilla1, qubit3, qubit4)
    circ.crx(-np.pi/2.0, ancilla1, qubit3)
    circ.crx(np.pi/2.0, ancilla1, qubit4)
    circ.cu1(np.pi/2.0, ancilla1, qubit3)

    return circ

def first_ord_exp_sigma_double_ex_simplified(u, v, circ, ancilla1, qubit1, qubit2, qubit3, qubit4):
    #YXXX
    circ.crx(np.pi/2.0, ancilla1, qubit1)
    circ.ch(ancilla1, qubit2)
    circ.ch(ancilla1, qubit3)
    circ.ch(ancilla1, qubit4)
    circ.rccx(ancilla1, qubit1, qubit2)
    circ.rccx(ancilla1, qubit2, qubit3)
    circ.rccx(ancilla1, qubit3, qubit4)
    circ.crz((-2.0*(lamdoub_val(u,v)-lam12_val(u,v)**2)).real, ancilla1, qubit4)
    circ.rccx(ancilla1, qubit3, qubit4)
    circ.rccx(ancilla1, qubit2, qubit3)
    circ.rccx(ancilla1, qubit1, qubit2)
    circ.crx(-np.pi/2.0, ancilla1, qubit1)
    circ.ch(ancilla1, qubit2)
    circ.ch(ancilla1, qubit3)
    circ.ch(ancilla1, qubit4)
    return circ

def full_trotter_step(step_num, trot_steps,is_Ua_term, u, v, ecc, circ, ancilla1, qubit1, qubit2, qubit3, qubit4):
    if is_Ua_term==True:
        u=-u
        v=-v
    else:
        pass
    delta_t = dt/trot_steps
    for n in range(0,step_num):
        for j in range(0, trot_steps):
            circ.ccx(ancilla1, qubit1, qubit3)
            circ.crz(twopi*(u/4.0)*delta_t, ancilla1, qubit3)
            circ.ccx(ancilla1, qubit1, qubit3)

            circ.rz((twopi*(u/4.0 + ecc)*delta_t).real,ancilla1)

            circ.ccx(ancilla1, qubit1, qubit2)
            circ.crx(twopi*v*delta_t, ancilla1, qubit1)
            circ.ch(ancilla1, qubit1)
            circ.ccx(ancilla1, qubit1, qubit2)
            circ.cu1(np.pi/2.0, ancilla1, qubit1)
            circ.ch(ancilla1, qubit1)
            circ.crz(-twopi*v*delta_t, ancilla1, qubit2)
            circ.ccx(ancilla1, qubit1, qubit2)
            circ.crx(-np.pi/2.0, ancilla1, qubit1)
            circ.crx(np.pi/2.0, ancilla1, qubit2)

            circ.ccx(ancilla1, qubit3, qubit4)
            circ.crx(twopi*v*delta_t, ancilla1, qubit3)
            circ.ch(ancilla1, qubit3)
            circ.ccx(ancilla1, qubit3, qubit4)
            circ.cu1(np.pi/2.0, ancilla1, qubit3)
            circ.ch(ancilla1, qubit3)
            circ.crz(-twopi*v*delta_t, ancilla1, qubit4)
            circ.ccx(ancilla1, qubit3, qubit4)
            circ.crx(-np.pi/2.0, ancilla1, qubit3)
            circ.crx(np.pi/2.0, ancilla1, qubit4)

            circ.ccx(ancilla1, qubit1, qubit3)
            circ.crz(twopi*(u/4.0)*delta_t, ancilla1, qubit3)
            circ.ccx(ancilla1, qubit1, qubit3)

    return circ

def sing_ex_trotter_step(step_num, u, v, circ, ancilla1, qubit1, qubit2, qubit3, qubit4):
    for n in range(0, step_num):

        circ.ccx(ancilla1, qubit1, qubit2)
        circ.crx(twopi*v*dt, ancilla1, qubit1)
        circ.ch(ancilla1, qubit1)
        circ.ccx(ancilla1, qubit1, qubit2)
        circ.cu1(np.pi/2.0, ancilla1, qubit1)
        circ.ch(ancilla1, qubit1)
        circ.crz(-twopi*v*dt, ancilla1, qubit2)
        circ.ccx(ancilla1, qubit1, qubit2)
        circ.crx(-np.pi/2.0, ancilla1, qubit1)
        circ.crx(np.pi/2.0, ancilla1, qubit2)

        circ.ccx(ancilla1, qubit3, qubit4)
        circ.crx(twopi*v*dt, ancilla1, qubit3)
        circ.ch(ancilla1, qubit3)
        circ.ccx(ancilla1, qubit3, qubit4)
        circ.cu1(np.pi/2.0, ancilla1, qubit3)
        circ.ch(ancilla1, qubit3)
        circ.crz(-twopi*v*dt, ancilla1, qubit4)
        circ.ccx(ancilla1, qubit3, qubit4)
        circ.crx(-np.pi/2.0, ancilla1, qubit3)
        circ.crx(np.pi/2.0, ancilla1, qubit4)

    return circ
