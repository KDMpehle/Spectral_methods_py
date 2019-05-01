# Python scripts for chapter six of Trefethen's 'spectral methods
# in matlab', translated by Khaya Mpehle
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt


program_number = 12 # select which program to be run

def cheb(N):
    # this function returns the Chebychev differentiation matrix
    if N==0:
        D = 0
        x =1
        return
    NN = np.array(range(0,N+1)) # array of grid point indices
    x= np.cos(np.pi*NN/N) # the chebyshev points 
    x.shape = (N+1,1) # Reshape into a column vector
    signs = np.power(-1,NN) # array of signs of Chebychev elements
    signs.reshape((N+1,1)) #reshape into a column vector
    c= np.concatenate([[2],np.ones(N-1),[2]]).T*(signs)
    c= c.reshape((N+1,1))# c is a special vector to help define D
    X= repmat(x,1,N+1) # grid points replicated to assist differencing
    dX= X-(X.T) # for difference between points
    D = (c.dot((1./c).T))/(dX+np.eye(N+1)) # relation for off-diagionals
    D = D- np.diag(np.sum(D.T,axis =0)) #adjust diagonals.
    return D,x 
                      
if program_number == 11:# Differentiating a smooth function with Chebychev matrix.
    xx = np.linspace(-1,1,100)
    uu = np.exp(xx)*np.sin(5*xx)
    for N in range(10,30,10):
        (D,x) = cheb(N) # get Chebychev matrix and points.
        u = np.exp(x)*np.sin(5*x)
        plt.subplot(2,2,2*(N/10)-1)
        plt.plot(x,u,'o',ms = 8)
        plt.plot(xx,uu,color = 'blue')
        plt.title('u(x), N={:d}'.format(N))
        error = D.dot(u)-np.exp(x)*(np.sin(5*x) + 5*np.cos(5*x))
        plt.subplot(2,2,2*(N/10))
        plt.plot(x,error,'-o',ms = 8)
        plt.title(" error in u'(x), N ={:d}".format(N))
    plt.show()
elif program_number == 12:# spectral differentiation accuracy on various functions.
    Nmax = 50
    E = np.zeros((4,Nmax)) # accuracy varies with N, for four functions.
    for N in range(Nmax):
        (D,x) = cheb(N+1) # get Chebychev matrix and points.
        v, vprime = np.abs(x)**3, 3*x*np.abs(x)
        E[0,N]  = np.linalg.norm(D.dot(v) - vprime,np.inf)
        v = np.exp(-x**-2)
        vprime = 2*(x**-3)*v # c-infinity function, npt analytic!
        E[1,N] = np.linalg.norm(D.dot(v) -vprime,np.inf)
        v = 1./(1+x**2) # an analytic function in [-1,1]
        vprime = -2*x*(v**2)
        E[2,N] = np.linalg.norm(D.dot(v) - vprime, np.inf)
        v ,vprime = x**10, 10*x**9  # polynomial.
        E[3,N] = np.linalg.norm(D.dot(v) - vprime, np.inf)
    #plot the result
    title = [r'$|x|^3$',r'$\exp(-x^2)$', r'$\frac{1}{1+x^2}$',r'$x^{10}$']
    for i in range(4):
        plt.subplot(2,2,i)
        plt.semilogy(np.array(range(1,Nmax+1)),E[i,:],'o',ms = 8)
        plt.plot(np.array(range(1,Nmax+1)),E[i,:], color = 'blue')
        plt.xlabel('N')
        plt.ylabel(r' Error ($\infty$-norm)')
        plt.title(title[i])
    plt.tight_layout()
    plt.show()
    
    
    
