# Python scripts for chapter six of Trefethen's 'spectral methods
# in matlab', translated by Khaya Mpehle
import numpy as np
from numpy.matlib import repmat

program_number = 11 # select which program to be run

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
                      
# finish the rest of the programs in this chapter. 
