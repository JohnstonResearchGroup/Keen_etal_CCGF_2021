import numpy as np
import math, cmath
from scipy.linalg import expm
from Pauli_matrices import *
from sympy.functions.special.tensor_functions import KroneckerDelta

tau_check = False

np.set_printoptions(linewidth=np.inf)

c1dag = np.matrix(np.kron(np.kron(sigminus, I2), np.kron(I2, I2)))
c2dag = np.matrix(np.kron(np.kron(Z, sigminus), np.kron(I2, I2)))
c3dag = np.matrix(np.kron(np.kron(Z, Z), np.kron(sigminus, I2)))
c4dag = np.matrix(np.kron(np.kron(Z, Z), np.kron(Z, sigminus)))

c1 = c1dag.getH()
c2 = c2dag.getH()
c3 = c3dag.getH()
c4 = c4dag.getH()

def spec_form(omega, alpha1, omega1, alpha2, omega2, eta):
    return alpha1/(omega + complex(0.0, eta) - omega1) + alpha2/(omega + complex(0.0, eta) - omega2) + alpha1/(omega + complex(0.0, eta) + omega1) + alpha2/(omega + complex(0.0, eta) + omega2)

#All of these functions can be converted to their sub and superscript
#expression by observing the following ordering:
#upper indices first, then lower.
#In x values, the value of p is the last number, otherwise
#they follow the same ordering as above. Similar for the S's.

def commutator(A, B):
    return A@B - B@A

def Power(x, y):
    return x**y

def t21_val(u, v):
    return 0.125*(u - np.sqrt(u*u + 64.0*v*v))/v

def t34_val(u, v):
    return 0.125*(u - np.sqrt(u*u + 64.0*v*v))/v

def tdoub_val(u, v):
    return -0.25*u*t21_val(u,v)/v

'''
Sm/Rd amplitudes and operators
'''
def sm3_21_val(omega, u, v, t):
    return 0.75 * u * t / (omega - 0.5 * u - v * t)

def sm3_34_val(omega, u, v, t):
    return -t34_val(u, v)

def sm3_3241_val(omega, u, v, t):
    return -tdoub_val(u, v)

def Sm_op(omega, u, v, t):
    sm21   = sm3_21_val(omega, u, v, t)
    sm34   = sm3_34_val(omega, u, v, t)
    sm3241 = sm3_3241_val(omega, u, v, t)
    return sm21*c1dag@c2 + sm34*c4dag@c3 + sm3241*c4dag@c1dag@c2@c3

def rd1_21_val(omega, u, v, t):
    return -t21_val(u, v)

def rd1_34_val(omega, u, v, t):
    return -0.75 * u * t / (omega + 0.5 * u + v * t)

def rd1_3241_val(omega, u, v, t):
    return -tdoub_val(u, v)

def Rd_op(omega, u, v, t):
    rd21   = rd1_21_val(omega, u, v, t)
    rd34   = rd1_34_val(omega, u, v, t)
    rd3241 = rd1_3241_val(omega, u, v, t)
    return rd21*c1dag@c2 + rd34*c4dag@c3 + rd3241*c4dag@c1dag@c2@c3

def x33_val(omega, u, v, t):
    return 1.0 / (omega + v * t + 0.75 * u * v * t / (omega - 0.5 * u - v * t))

def x2313_val(omega, u, v, broadening):
    return ((u*(-0.03125*u + 0.03125*cmath.sqrt(Power(u,2) + 64.*Power(v,2))))/
       (v*(1.*Power(v,2) - complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*broadening - 0.3333333333333333*Power(broadening,2) +
       0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega - complex(0.,0.6666666666666666)*broadening*omega + 0.3333333333333333*Power(omega,2))))

def lam12_val(u, v):
    return (-4.0*v)/np.sqrt(u*u + 64.0*v*v)

def lam43_val(u, v):
    return (-4.0*v)/np.sqrt(u*u + 64.0*v*v)

def lamdoub_val(u, v):
    return lam43_val(u,v)/(2.0*t21_val(u,v))

def T_op(u, v):
    return ((t21_val(u, v)*c1dag@c2) + (t34_val(u, v)*c4dag@c3) + (tdoub_val(u, v)*c4dag@c1dag@c2@c3))

def lambda_op(u,v):
    return lam12_val(u, v)*c2dag@c1 + lam43_val(u, v)*c3dag@c4 + (lamdoub_val(u,v)*c3dag@c2dag@c1@c4)

def y11_val(omega, u, v, t):
    return 1.0 / (omega - v * t + 0.75 * u * v * t / (omega + 0.5 * u + v * t))

def y3141_val(omega, u, v, broadening):
    return ((u*(-0.03125*u + 0.03125*cmath.sqrt(Power(u,2) + 64.*Power(v,2))))/
   (v*(1.*Power(v,2) + complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening - 0.3333333333333333*Power(-1.0*broadening,2) -
       0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega - complex(0.,0.6666666666666666)*-1.0*broadening*omega + 0.3333333333333333*Power(omega,2)
       )))

def y12_val(omega, u, v, broadening):
    return ((v*(0.3333333333333333*u - 1.*cmath.sqrt(Power(u,2) + 64.*Power(v,2)) - complex(0.,2.6666666666666665)*-1.0*broadening + 2.6666666666666665*omega))/
   ((u + cmath.sqrt(Power(u,2) + 64.*Power(v,2)))*(1.*Power(v,2) + complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening -
       0.3333333333333333*Power(-1.0*broadening,2) - 0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega -
       complex(0.,0.6666666666666666)*-1.0*broadening*omega + 0.3333333333333333*Power(omega,2))))

def y3142_val(omega, u, v, broadening):
    return ((0.3333333333333333*u*(1.*cmath.sqrt(Power(u,2) + 64.*Power(v,2)) + complex(0.,2.0000000000000004)*-1.0*broadening - 2.0000000000000004*omega))/
   ((u + cmath.sqrt(Power(u,2) + 64.*Power(v,2)))*(1.*Power(v,2) + complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening -
       0.3333333333333333*Power(-1.0*broadening,2) - 0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega -
       complex(0.,0.6666666666666666)*-1.0*broadening*omega + 0.3333333333333333*Power(omega,2))))

def y43_val(omega, u, v, broadening):
    return ((-1.*u*Power(v,2) + Power(v,2)*(-1.*cmath.sqrt(Power(u,2) + 64.*Power(v,2)) - complex(0.,2.6666666666666665)*-1.0*broadening + 2.6666666666666665*omega))/
   (v*(u + cmath.sqrt(Power(u,2) + 64.*Power(v,2)))*(1.*Power(v,2) +
       complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening - 0.3333333333333333*Power(-1.0*broadening,2) -
       0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega - complex(0.,0.6666666666666666)*-1.0*broadening*omega + 0.3333333333333333*Power(omega,2)
       )))

def y2143_val(omega, u, v, broadening):
    return ((u*(complex(0.,-0.6666666666666667)*-1.0*broadening + 0.6666666666666667*omega))/
   ((u + cmath.sqrt(Power(u,2) + 64.*Power(v,2)))*(1.*Power(v,2) + complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening -
       0.3333333333333333*Power(-1.0*broadening,2) - 0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega -
       complex(0.,0.6666666666666666)*-1.0*broadening*omega + 0.3333333333333333*Power(omega,2))))

def y44_val(omega, u, v, broadening):
    return ((complex(0.,0.25)*u + complex(0.,0.75)*cmath.sqrt(Power(u,2) + 64.*Power(v,2)) - 2.*-1.0*broadening - complex(0.,2.)*omega)/
   (complex(0.,-6.)*Power(v,2) + 1.*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening + complex(0.,2.)*Power(-1.0*broadening,2) +
     complex(0.,1.)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega - 4.*-1.0*broadening*omega - complex(0.,2.)*Power(omega,2)))

def y2144_val(omega, u, v, broadening):
    return ((u*(-0.010416666666666668*u + 0.010416666666666668*cmath.sqrt(Power(u,2) + 64.*Power(v,2))))/
   (v*(1.*Power(v,2) + complex(0.,0.16666666666666666)*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*-1.0*broadening - 0.3333333333333333*Power(-1.0*broadening,2) -
       0.16666666666666666*cmath.sqrt(Power(u,2) + 64.*Power(v,2))*omega - complex(0.,0.6666666666666666)*-1.0*broadening*omega + 0.3333333333333333*Power(omega,2)
       )))

def sigma_op(u, v):
    sigma_operator = (lam12_val(u, v)*(c2dag@c1) + lam43_val(u, v)*(c3dag@c4) + (lamdoub_val(u, v) - lam12_val(u, v)*lam43_val(u, v))*c2dag@c3dag@c4@c1 - np.conj(lam12_val(u, v))*c1dag@c2 - np.conj(lam43_val(u, v))*c4dag@c3 - (np.conj(lamdoub_val(u, v)) - np.conj(lam12_val(u, v)*lam43_val(u, v)))*c1dag@c4dag@c3@c2)
    #sigma_operator = (lam1_Bo(u, v)*(c2dag@c1 + c3dag@c4) + 0.25*(lam2_Bo(u, v) - lam1_Bo(u, v)*np.conj(lam1_Bo(u, v)))*c2dag@c3dag@c4@c1
    #    - np.conj(lam1_Bo(u, v))*c1dag@c2 - np.conj(lam1_Bo(u, v))*c4dag@c3 - 0.25*(np.conj(lam2_Bo(u, v)) - np.conj(lam1_Bo(u, v)*lam1_Bo(u, v)))*c1dag@c4dag@c3@c2)
    return sigma_operator

def taum(omega, u, v, broadening, m):
    if m==2:
        return 1.0/x22_val(omega, u, v, broadening)
    elif m==3:
        return 1.0/x33_val(omega, u, v, broadening)
    else:
        raise ValueError('Invalid m')

def taud(omega, u, v, broadening, d):
    if d==1:
        return 1.0/y11_val(omega, u, v, broadening)
    elif d==4:
        return 1.0/y44_val(omega, u, v, broadening)
    else:
        raise ValueError('Invalid d')

def S212_val(omega, u, v, broadening):
    if tau_check==True and abs(taum(omega, u, v, broadening, 2))>10.0:
        #print('help1')
        return 0.0
    else:
        return -t21_val(u,v)

def S342_val(omega, u, v, broadening):
    if tau_check==True and abs(taum(omega, u, v, broadening, 2))>10.0:
        #print('help')
        return 0.0
    else:
        S342_vals = (taum(omega, u, v, -broadening, 2)*x2342_val(omega, u, v, broadening))
        return S342_vals

#x^{ij}_a = -x^{ji}_a
def S213_val(omega, u, v, broadening):
    if tau_check==True and abs(taum(omega, u, v, broadening, 3))>10.0:
        #print('help2')
        return 0.0
    else:
        return (taum(omega, u, v, broadening, 3)*-1.0*x2313_val(omega, u, v, broadening))
        #return (taum(omega, u, v, broadening, 3)*-1.0*x2_Bo(omega, u, v, broadening, E0N))
    #return 0.0

def S343_val(omega, u, v, broadening):
    if tau_check==True and abs(taum(omega, u, v, broadening, 2))>10.0:
        #print('help')
        return 0.0
    else:
        return -t34_val(u,v)

def S23142_val(omega, u, v, broadening):
    if tau_check==True and abs(taum(omega, u, v, broadening, 2))>10.0:
        return 0.0
    else:
        S23142_vals = (-1.0*tdoub_val(u, v) + taum(omega, u, v, broadening, 2)*(-1.0*t21_val(u,v)*x2342_val(omega, u, v, broadening)) - S212_val(omega, u, v, broadening)*S342_val(omega, u, v, broadening))
        return S23142_vals

def S23143_val(omega, u, v, broadening):
    #if tau_check==True and abs(taum(omega, u, v, broadening, 3))>10.0:
        #print('help3')
        return -tdoub_val(u,v)
        '''
    return (taum(omega, u, v, broadening, 3)*(t34_val(u,v)*x2313_val(omega, u, v, broadening)
     - tdoub_val(u, v)*x33_val(omega, u, v, broadening)) + taum(omega, u, v, broadening,3)**2 * t34_val(u, v)
      * x33_val(omega, u, v, broadening) * x2313_val(omega, u, v, broadening))
      '''
    #return np.zeros(len(omega), dtype=complex)
    #else:
        #return -tdoub_val(u,v)+
    #    return taum(omega, u, v, broadening, 3)*t34_val(u,v)*x2313_val(omega, u, v, broadening)- S213_val(omega, u, v, broadening)*S343_val(omega, u, v, broadening)


def R211_val(omega, u, v, broadening):
    if tau_check==True and abs(taud(omega, u, v, broadening, 1))>10.0:
        #print('help4')
        return 0.0
    else:
        return (-t21_val(u, v))

def R341_val(omega, u, v, broadening):
    if tau_check==True and abs(taud(omega, u, v, broadening, 1))>10.0:
        #print('help')
        return 0.0
    else:
        return (taud(omega, u, v, broadening, 1)*(y3141_val(omega, u, v, broadening)))
        #return (taud(omega, u, v, broadening, 1)*(y2_Bo(omega, u, v, broadening, E0N)))

#y^i_{ab} = -y^i_{ba}
def R214_val(omega, u, v, broadening):
    if tau_check==True and abs(taud(omega, u, v, broadening, 4))>10.0:
        #print('help')
        return 0.0
    else:
        return (taud(omega, u, v, broadening, 4)*-1.0*y2144_val(omega, u, v, broadening))
    #return 0.0

def R344_val(omega, u, v, broadening):
    if tau_check==True and abs(taud(omega, u, v, broadening, 4))>10.0:
        #print('help')
        return 0.0
    else:
        return (-t34_val(u, v))

def R23141_val(omega, u, v, broadening):
    #if tau_check==True and abs(taud(omega, u, v, broadening, 1))>10.0:
        #print('help5')
    #return 0.0
    #return np.zeros(len(omega))
    #else:
    return (-tdoub_val(u, v) + taud(omega, u, v, broadening, 1)*(-t21_val(u,v)*y3141_val(omega, u, v, broadening)) - R211_val(omega, u, v, broadening)*R341_val(omega, u, v, broadening))

def R23144_val(omega, u, v, broadening):
    if tau_check==True and abs(taud(omega, u, v, broadening, 4))>10.0:
        #print('help')
        return 0.0
    else:
        return (-tdoub_val(u, v) + taud(omega, u, v, broadening, 4)*(-1.0*t34_val(u, v)*y2144_val(omega, u, v, broadening)) - R214_val(omega, u, v, broadening)*R344_val(omega, u, v, broadening))

def deltam2_op(omega, u, v, broadening):
    return (S212_val(omega, u, v, broadening)*c1dag@c2 -
        (np.conjugate(np.transpose(S212_val(omega, u, v, broadening)*c1dag@c2
        + S342_val(omega, u, v, broadening)*c4dag@c3))) + S23142_val(omega, u, v, broadening)*c1dag@c4dag@c3@c2 -
        np.conjugate(np.transpose(S23142_val(omega, u, v, broadening)*c1dag@c4dag@c3@c2)))

def deltam3_op(omega, u, v, broadening):
    return (S213_val(omega, u, v, broadening)*c1dag@c2 + S343_val(omega, u, v, broadening)*c4dag@c3 -
            np.conjugate(S213_val(omega, u, v, broadening))*c2dag@c1 - np.conjugate(S343_val(omega, u, v, broadening))*c3dag@c4
            + S23143_val(omega, u, v, broadening)*c1dag@c4dag@c3@c2 - np.conjugate(S23143_val(omega, u, v, broadening))*c2dag@c3dag@c4@c1)

def deltad1_op(omega, u, v, broadening):
    return (R211_val(omega, u, v, broadening)*c1dag@c2 + R341_val(omega, u, v, broadening)*c4dag@c3 -
            np.conjugate(R211_val(omega, u, v, broadening))*c2dag@c1 - np.conjugate(R341_val(omega,u,v, broadening))*c3dag@c4
            + 0.25*R23141_val(omega, u, v, broadening)*c1dag@c4dag@c3@c2 - 0.25*np.conjugate(R23141_val(omega, u, v, broadening))*
            c2dag@c3dag@c4@c1)

def deltad4_op(omega, u, v, broadening):
    return (R214_val(omega, u, v, broadening)*c1dag@c2 + R344_val(omega, u, v, broadening)*c4dag@c3 -
            np.conjugate(R214_val(omega, u, v, broadening))*c2dag@c1 - np.conjugate(R344_val(omega,u,v, broadening))*c3dag@c4
            + 0.25*R23144_val(omega, u, v, broadening)*c1dag@c4dag@c3@c2 - 0.25*np.conjugate(R23144_val(omega, u, v, broadening))*
            c2dag@c3dag@c4@c1)

def exp_deltam2_op_expval(trial_vec, omega, u, v, broadening):
    exp_deltam2_op_expvals = np.zeros(len(omega), dtype=complex)
    for w in np.arange(len(omega)):
        exp_deltam2_op_expvals[w] = np.conjugate(trial_vec).transpose()@expm(deltam2_op(omega[w], u, v, broadening))@trial_vec
        '''
        (np.conjugate(trial_vec).transpose()@expm(S212_val(omega[w], u, v, broadening)*c1dag@c2 -
        (np.conjugate(np.transpose(S212_val(omega[w], u, v, broadening)*c1dag@c2
        + S342_val(omega[w], u, v, broadening)*c4dag@c3))) + S23142_val(omega[w], u, v, broadening)*c1dag@c4dag@c3@c2 -
        np.conjugate(np.transpose(S23142_val(omega[w], u, v, broadening)*c1dag@c4dag@c3@c2)))@trial_vec)
        '''
    return exp_deltam2_op_expvals

def exp_deltam3_op_expval(trial_vec, omega, u, v, broadening):
    exp_deltam3_op_expvals = np.zeros(len(omega), dtype=complex)
    for w in np.arange(len(omega)):
        exp_deltam3_op_expvals[w] = np.conjugate(trial_vec).transpose()@expm(deltam3_op(omega[w], u, v, broadening))@trial_vec
        '''
        (np.conjugate(trial_vec).transpose()@expm(S213_val(omega[w], u, v, broadening)*c1dag@c2 + S343_val(omega[w], u, v, broadening)*c4dag@c3 -
            np.conjugate(S213_val(omega[w], u, v, broadening))*c2dag@c1 - np.conjugate(S343_val(omega[w], u, v, broadening))*c3dag@c4
            + S23143_val(omega[w], u, v, broadening)*c1dag@c4dag@c3@c2 - np.conjugate(S23143_val(omega[w], u, v, broadening))*c2dag@c3dag@c4@c1)@trial_vec)
        '''
    return exp_deltam3_op_expvals

def exp_deltad1_op_expval(trial_vec, omega, u, v, broadening):
    exp_deltad1_op_expvals = np.zeros(len(omega), dtype=complex)
    for w in np.arange(len(omega)):
        exp_deltad1_op_expvals[w] = np.conjugate(trial_vec).transpose()@expm(deltad1_op(omega[w], u, v, broadening))@trial_vec
        '''
        (np.conjugate(trial_vec).transpose()@expm(R211_val(omega[w], u, v, broadening)*c1dag@c2 + R341_val(omega[w], u, v, broadening)*c4dag@c3 -
            np.conjugate(R211_val(omega[w], u, v, broadening))*c2dag@c1 - np.conjugate(R341_val(omega[w],u,v, broadening))*c3dag@c4
            + 0.25*R23141_val(omega[w], u, v, broadening)*c1dag@c4dag@c3@c2 - 0.25*np.conjugate(R23141_val(omega[w], u, v, broadening))*
            c2dag@c3dag@c4@c1)@trial_vec)
        '''
    return exp_deltad1_op_expvals

def exp_deltad4_op_expval(trial_vec, omega, u, v, broadening):
    exp_deltad4_op_expvals = np.zeros(len(omega), dtype=complex)
    for w in np.arange(len(omega)):
        exp_deltad4_op_expvals[w] = np.conjugate(trial_vec).transpose()@expm(deltad4_op(omega[w], u, v, broadening))@trial_vec
        '''
        (np.conjugate(trial_vec).transpose()@expm(R214_val(omega[w], u, v, broadening)*c1dag@c2 + R344_val(omega[w], u, v, broadening)*c4dag@c3 -
            np.conjugate(R214_val(omega[w], u, v, broadening))*c2dag@c1 - np.conjugate(R344_val(omega[w],u,v, broadening))*c3dag@c4
            + 0.25*R23144_val(omega[w], u, v, broadening)*c1dag@c4dag@c3@c2 - 0.25*np.conjugate(R23144_val(omega[w], u, v, broadening))*
            c2dag@c3dag@c4@c1)@trial_vec)
        '''
    return exp_deltad4_op_expvals

def inv_sigma_op_val(trial_vec, u, v):
    return 1.0/(np.conj(trial_vec).transpose()@expm(sigma_op(u, v))@trial_vec)

def inv_expec_deltam3_op_val(trial_vec, omega, u, v, broadening):
    return np.exp(-np.log(exp_deltam3_op_expval(trial_vec, omega, u, v, broadening)))

def A3(omega, u, v, broadening):
    return np.sqrt(1.0 + np.conjugate(S213_val(omega, u, v, broadening))*S213_val(omega, u, v, broadening) + np.conjugate(S343_val(omega, u, v, broadening))*S343_val(omega, u, v, broadening) + 0.25*np.conjugate(S23143_val(omega, u, v, broadening))*S23143_val(omega, u, v, broadening))

def A1(omega, u, v, broadening):
    return np.sqrt(1.0 + np.conjugate(R211_val(omega, u, v, broadening))*R211_val(omega, u, v, broadening) + np.conjugate(R341_val(omega, u, v, broadening))*R341_val(omega, u, v, broadening) + 0.25*np.conjugate(R23141_val(omega, u, v, broadening))*R23141_val(omega, u, v, broadening))

def inv_expec_deltad1_op_val(trial_vec, omega, u, v, broadening):
    return np.exp(-np.log(exp_deltad1_op_expval(trial_vec, omega, u, v, broadening)))

def both_expec_vals_lesser(trial_vec, omega, u, v, broadening):
    both_expec_vals_lesser_vals = np.zeros(len(omega), dtype=complex)
    for w in np.arange(len(omega)):
        both_expec_vals_lesser_vals[w] = np.conj(trial_vec).transpose()@expm(sigma_op(u, v))@expm(deltad1_op(omega[w], u, v, broadening))@trial_vec
    return both_expec_vals_lesser_vals

def both_expec_vals_greater(trial_vec, omega, u, v, broadening):
    both_expec_vals_greater_vals = np.zeros(len(omega), dtype=complex)
    for w in np.arange(len(omega)):
        both_expec_vals_greater_vals[w] = np.conj(trial_vec).transpose()@expm(sigma_op(u, v))@expm(deltam3_op(omega[w], u, v, broadening))@trial_vec
    return both_expec_vals_greater_vals

def g(omega, u, v, E0N, broadening):
    return (1.0 + (3.0*u)/(4.0*(omega + complex(0.0,broadening) + E0N - u/2.0 - v*t21_val(u,v))))*t21_val(u,v)

def h(omega, u, v, E0N, broadening):
    return (1.0 - (3.0*u)/(4.0*(omega - complex(0.0,broadening) - E0N + u/2.0 + v*t21_val(u,v))))*t34_val(u,v)

#print(np.matrix(c1))

'''
def equations(p, v):
    t21, t34, tdoub = p
    return (4.0*v + tdoub*v + t21*16.0 - 4.0*t21*t21*v, 4.0*v + tdoub*v + t34*16.0 - 4.0*t34*t34*v, -128.0*t21*t34 - 8.0*tdoub*t21*v - 8.0*tdoub*t34*v)
'''
