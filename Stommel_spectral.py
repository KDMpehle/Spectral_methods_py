'''
A script to solve the Stommel streamfunction problem
for ocean gyre modelling in the beta-plane approximation

r(psi_xx + psi_yy) + betapsi_x = tau_0 sin(pi*y/b)

r = bottom drag of the ocean
tau_0= amplitude of surface wind stress.
beta = beta parameter.
'''
import numpy as np
import matplotlib.pyplot as plt
from chapter_six import cheb
from scipy.linalg import solve
#physical parameters
e = 0.04 # epsilon = r/(abeta), a meridional length scale.
(a,b) = (0,1) #[0,1] x [0,1]
#Pseudo-spectral method.
N= 32 # number of Chebycheff points.
(D,x) = cheb(N) # get the Chebychev grid.
D2= D.dot(D) # second derivative matrix.
D1 = D # first derivative matrix.
D2[0,:],D1[0,:]=np.concatenate([[1],np.zeros(N)]),np.concatenate([[1],np.zeros(N)])
D2[N,:], D1[N,:]= np.concatenate([np.zeros(N),[1]]),np.concatenate([np.zeros(N),[1]])
J = 2/(a-b) # scale factor for derivative
D2 = J**2*D2
D1 = J*D1 # first derivative.
X,Y = np.meshgrid(x,x) # mesh grid
xm, ym = X.flatten(), Y.flatten() # flatten grids to vectors.
scale_y = (a-b)*(ym +(a+b)/(a-b))/2 # physical spatial variables. 
scale_x = (a-b)*(xm + (a+b)/(a-b))/2
#find the boundary and apply Dirichlet
boundary = np.nonzero(((np.abs(xm) ==1) | (np.abs(ym) ==1)))[0]
f= np.sin(np.pi*scale_y) # wind stress function.
f[boundary] = 0
I = np.eye(N+1) # 2-D Kronecker product.
L = e*np.kron(I,D2) + e*np.kron(D2,I) + np.kron(I,D1)
u = solve(L,f) # solve the convection-diffusion.
uu = np.zeros((N+1,N+1)) # hold the solution in 2D array.
uu = u.reshape((N+1,N+1))
scale_y = (a-b)*(Y +(a+b)/(a-b))/2 # physical spatial variables. 
scale_x = (a-b)*(X + (a+b)/(a-b))/2
plt.contour(scale_x,scale_y,uu,10,colors= 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Stommel streamfunction: $\epsilon = {:.2f}$'.format(e))
#plt.savefig('StommelST.pdf')
plt.show()

