# Chapter one of spectral methods by Trefethen
import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
program_val = 2
if program_val == 1: #Program one: fourth order finite difference convergence 
    N_vals = [2**k for k in range (3,12)]
    fig = plt.figure(figsize= (16,8))
    ax = plt.axes()
    errorvec = [] # list to contain error
    for N in N_vals:
        h= 2*np.pi/N
        xx = np.linspace(-np.pi+h,np.pi,N)
        u = np.exp(np.sin(xx)) 
        uprime = np.cos(xx)*u # the exact derivative
        #construct the fourth-order differentiation matrix
        e = np.ones(N)
        data = np.array([2*e/3, -e/12,e/12,-2*e/3])
        diags = np.array([1,2,N-2,N-1])
        D = sparse.spdiags(data,diags,N,N)
        Dprime = D.T # take the transpose
        D = (D-Dprime)/h 
        #plot the maxmum difference error
        error = np.linalg.norm(D.dot(u)-uprime,np.inf) #D.dot(u) finite difference derivative
        errorvec.append(error)
    ax.loglog(N_vals,errorvec,'--x',mew = 4,ms = 8)
    ax.set_xlabel('N',fontsize= 14)
    ax.set_ylabel(r'Error in the Infinity norm',fontsize = 14)
    title = r'4th order finite difference of $\exp(\sin(x))$'
    ax.set_title(title,fontsize = 18)
    plt.show()
elif program_val == 2:
    #program two: spectral convergence
    N_vals = [2*k for k in range(1,20)]
    errorvec=[]
    fig = plt.figure(figsize = (16,8))
    ax = plt.axes()
    for k in range(len(N_vals)):
        N = N_vals[k]
        h= 2*np.pi/N
        xx = np.linspace(-np.pi+h,np.pi,N)
        u= np.exp(np.sin(xx))
        uprime = np.cos(xx)*u
        #construct the spectral differentiation matrix
        NN = np.array(range(1,N)) 
        cols= np.concatenate([[0],0.5*np.power(-1,NN)/np.tan(NN*h/2)])
        rows = np.concatenate([[cols[0]],cols[:0:-1]])
        D= toeplitz(cols,rows) #construct as toeplitz matrix.
        error = np.linalg.norm(D.dot(u) - uprime,np.inf)
        errorvec.append(error)
    ax.loglog(N_vals,errorvec,'--x',color='k', mew =4,ms=8) #log-log plot
    title = " Error by spectral differentiation"
    ax.set_title(title,fontsize = 18)
    ax.set_xlabel("N",fontsize = 14)
    ax.set_ylabel("Infinity norm error", fontsize = 10)
    plt.show()          
