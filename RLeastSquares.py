# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:25:42 2020

Implementation of Privacy Preserving Recursive Least Squares Equations. The privacy is based on Real Number Secret sharing.

@author: kst
"""

import numpy as np
from RealNumberSecretSharing import RNS
import matplotlib.pyplot as plt

def MatVecProd(A,v):
    '''
    Computes securely the matrix vector product.
    Input: A, v is secret shared matrix and vector respectively.
    Outout: secret shared vector.
    '''
    I,J = np.shape(A)[:2]
    l = []
    for i in range(I):
        s = 0
        for j in range(J):
            s+= rns.mult(v[j], A[i,j])
        l.append(s)
    return np.array(l)

def VecMatProd(v,A):
    '''
    Computes securely the vector matrix product.
    Input: A, v is secret shared matrix and vector respectively.
    Outout: secret shared vector.
    '''
    I,J = np.shape(A)[:2]
    l = []
    for j in range(J):
        s = 0
        for i in range(I):
            s+= rns.mult(v[i], A[i,j])
        l.append(s)
    return np.array(l)

def VecVectProd(v1,v2):
    '''
    Computes securely the vector vector product.
    Input: v1 ( n x 1 ), and v2 ( 1 X n ) are secret shared vectors.
    Outout: secret shared matrix.
    '''
    I = len(v1)
    l = []
    for i in range(I):
        s = []
        for j in range(I):
            s.append(rns.mult(v2[i], v1[j]))
        l.append(s)
    return np.array(l)

def VectVecProd(v1,v2):
    '''
    Computes securely the vector vector product.
    Input: v1 ( 1 x n ), and v2 ( n X 1 ) are secret shared vectors.
    Outout: secret shared scalar.
    '''
    I = len(v1)
    s=0
    for i in range(I):
        s+= rns.mult(v1[i], v2[i])
    return s

def scalMatProd(s1,A):
    '''
    Computes securely the elementwise multiplication of a scalar and a matrix.
    Input: s1 is a secret shared scalar and A is a secret shared matrix.
    Outout: secret shared matrix.
    '''
    I,J = np.shape(A)[:2]
    l = []
    for i in range(I):
        s = []
        for j in range(J):
            s.append(rns.mult(s1,A[i,j]))
        l.append(s)
    return np.array(l)

def scalVecProd(s1, v):
    '''
    Computes securely the elementwise multiplication of a scalar and a vector.
    Input: s1 is a secret shared scalar and v is a secret shared vector.
    Outout: secret shared vector.
    '''
    R = [rns.mult(s1, i) for i in v]
    return np.array(R)


def recMat(A):
    '''
    Computes the plaint text of the matrix A.
    '''
    I,J = np.shape(A)[:2]
    R = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            R[i,j] = rns.rec(A[i,j])
    return R

def recVec(v):
    '''
    Computes the plaint text of the vector v.
    '''
    return np.array([rns.rec(i) for i in v]).reshape(len(v),1)




def RLS(dx, X, Y):
    '''
    Computes securely the recursive least squares equations.
    Input: dx is dimension of the system. X and Y are observations.
    Outout: secret shared estimate of parameters of the system.
    '''
    obs = len(Y)                    #number of observations
    Po = np.identity(dx)            #P0 matrix
    l = []
    #Secret share the P0 matrix
    for i in range(dx):
        l.append([rns.sharing(j) for j in Po[i,:] ])
    
    P = np.array(l)
    
    w = np.array([rns.sharing(0) for i in range(dx)]) #W0 parameter estimate
    w_open = []
    #secret share w0
    for i in range(obs):
        x = [rns.sharing(j) for j in X[i,:] ]          #Secret shared version of observations
        y = rns.sharing(Y[i])                   
        
        # Start with all parts of P
        d1 = 1 + VectVecProd(x, MatVecProd(P, x))
        d1_inv = rns.div(d1)
        d2 = VecVectProd( MatVecProd(P,x) , VecMatProd(x,P) )
        D = scalMatProd(d1_inv, d2)
        P = P - D
        
        # g
        g = MatVecProd(P,x)
        
        # e
        e = y - VectVecProd(x, w)
        
        #w
        w = w + scalVecProd(e,g)
        w_open.append(recVec(w))
    return recVec(w), w_open
        
 
#Main stuff:
np.random.seed(1)    
    
n=3                      #Number of parties
t = 1                    #privacy treshold
x = np.linspace(0.5,2,n) #Index numbers of parties

dx = 6                   #Dimension of system

rns = RNS(x,n,t)         #Class for the real number secret sharing scheme (all operations)

obs = 50                 #number of observations

X = np.random.normal(0,50,(obs, dx))       #Observations of input
#X = np.random.randint(0,5,(obs,dx))
Beta = np.random.normal(0,30, dx)          #True Parameters
#Beta = np.array([3.5,1.2,2.8,4.1,2.9,3.3])

y = np.array([np.dot(Beta, X[i,:]) for i in range(obs)])        #Observations of output
y = y + np.random.normal(0,5, obs)      #Add some noise to the output


w, w_open = RLS(dx, X, y)

print('True             ', '   estimat')
for i in range(dx):
    print('{}   {}'.format(Beta[i], w[i][0]))

mse = np.zeros(obs)
for i in range(obs):
    mse[i] = np.mean((Beta.reshape(dx,1) - np.array(w_open[i]))**2)

plt.plot(mse)

    