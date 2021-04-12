import numpy as np

#define Pauli matrices
I2 = [[1.0,0.0],
      [0.0,1.0]]

X = [[0.0,1.0],
     [1.0,0.0]]

Y = [[0.0, complex(0,-1.0)],
     [complex(0.0,1.0),0.0]]

Z = [[1.0,0.0],
     [0.0,-1.0]]

#print(np.matmul(X,X))
#print('\n')
#print(np.matmul(X,Y))
#print('\n')
#print(np.matmul(Y,X))
#print('\n')
#print(np.matmul(Z,Y))

#print(np.kron(np.kron(X,X),np.kron(X,X)))

Z1Z3 = np.kron(np.kron(Z,I2),np.kron(Z,I2))
#print(Z1Z3, '\n')
Z1 = np.kron(np.kron(Z,I2),np.kron(I2,I2))
Z2 = np.kron(np.kron(I2,Z),np.kron(I2,I2))
Z3 = np.kron(np.kron(I2,I2),np.kron(Z,I2))
Z4 = np.kron(np.kron(I2,I2),np.kron(I2,Z))

X1 = np.kron(np.kron(X,I2),np.kron(I2,I2))
X2 = np.kron(np.kron(I2,X),np.kron(I2,I2))
X3 = np.kron(np.kron(I2,I2),np.kron(X,I2))
X4 = np.kron(np.kron(I2,I2),np.kron(I2,X))

Y1 = np.kron(np.kron(Y,I2),np.kron(I2,I2))
Y2 = np.kron(np.kron(I2,Y),np.kron(I2,I2))
Y3 = np.kron(np.kron(I2,I2),np.kron(Y,I2))
Y4 = np.kron(np.kron(I2,I2),np.kron(I2,Y))

X1X2 = np.kron(np.kron(X,X),np.kron(I2,I2))
Y1Y2 = np.kron(np.kron(Y,Y),np.kron(I2,I2))
X3X4 = np.kron(np.kron(I2,I2),np.kron(X,X))
Y3Y4 = np.kron(np.kron(I2,I2),np.kron(Y,Y))

I4 = np.kron(I2, I2)

I  = np.eye(16)

sigplus = np.multiply(1/2, X + np.multiply(complex(0.0,1.0),Y))
sigminus = np.multiply(1/2, X - np.multiply(complex(0.0,1.0),Y))

#print(sigplus, '\n', sigminus)
