#Python scripts for chapter seven of Trefethen's 'spectral methods
# in matlab', translated by Khaya Mpehle
#chapter seven: boundary value problems
import numpy as np
import matplotlib.pyplot as plt
from chapter_six import cheb
from scipy.linalg import solve
from scipy import interpolate
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
program_number = 16 # select which program to be run 

if program_number == 13: # linear BVP u_xx = exp(4x), u(-1) = u(2)  = 0
    N = 16; # number of chebychev points 
    (D,x) = cheb(N) # chebychev points and differentiation matrix
    D2 = D.dot(D) # Square of D gives second derivative matrix
    D2 = D2[1:N,1:N] # impose homogeneous Dirichlet boundary conditions
    f = np.exp(4*x[1:N])
    u = solve(D2,f) # solves the Poisson equation
    u=np.insert(u,[0,len(u)],[0,0]) # add the zero values explicitly
    uext = (1./16)*(np.exp(4*x) -x*np.sinh(4)-np.cosh(4))#exact solution
    u= u.reshape(np.shape(uext))
    xx= np.linspace(-1,1,100) #interpolate to a finer grid.
    uu=np.polyval(np.polyfit(x.flatten(),u,N),xx) #interpolate solution to finer grid.
    fig = plt.figure(figsize = (16,8))
    ax = plt.axes()
    ax.plot(xx,uu,'-',color = 'b',label = 'Interpolated solution')
    ax.plot(x,u,'o',ms = 6,label = 'grid solution')
    plt.text(0.,-0.5,r'The maximum error at grid points is %.3e'%(norm(uext - u,np.inf)), fontsize = 16)
    plt.legend(loc = 'lower left')
    plt.show()

elif program_number == 14: # non-linear BVP u_xx =exp(u)  u(-1) = u(1)
    N= 16 # Number of Chebychev points.
    (D,x) = cheb(N) # chebychev differentiation matrix and points
    D2= D.dot(D) # second derivative
    D2=D2[1:N,1:N] # homogeneous Dirichlet boundary conditions
    tol =1e-6 # tolerance for iteration
    u= np.zeros(N-1)
    iters = 0 # counter for number of iterations
    delta = 1.# initialise delta.
    while delta > tol: # Commence fixed-point iteration.
        iters += 1
        f=np.exp(u)
        unew = solve(D2,f)
        delta = norm(np.abs(u-unew),np.inf)
        u=unew
    print(" solution converged after", iters, "iterations")
    u= np.insert(u,[0,len(u)],[0,0]) # attach the boundary conditions.
    xx = np.linspace(-1,1,100) # finer grid
    uu = np.polyval(np.polyfit(x.flatten(),u,N),xx) #interpolate solution.
    fig = plt.figure(figsize = (16,8))
    ax = plt.axes()
    ax.plot(x,u,'o', ms = 6, label = 'grid solution')
    ax.plot(xx,uu,'-', color = 'red', label = 'Interpolated solution')
    ax.set_xlabel('x')
    ax.set_title('Iterations= {:d},    u(0)={:f}'.format(iters,u[N/2]))
    plt.legend(loc = 'lower left')
    plt.show()

elif program_number == 15: # eigenvalue problem, with homogeneous Dirichlet.
    N = 36 # Number of Chebycheff points.
    (D,x) = cheb(N) # Chebycheff differentiation matrix and points
    D2 = D.dot(D) # second derivative.
    D2 = D2[1:N,1:N] # homogeneous Dirichlet boundary conditions.
    W,V = np.linalg.eig(D2) # W the eigenvalues, V the vectors.
    inds = np.argsort(-W) # indices to sort the eigenvalues
    Evals = np.real(W[inds]) # resort the eigenvalues.
    V = V[:,inds] #reindex the eigenvectors
    fig=plt.figure(figsize = (16,16))
    for k in range(5,35,5):
        plt.subplot(6,1,k/5).set_title(r' eig %d = %20.13f *4/$\pi^2$'%(k,Evals[k-1]*4/np.pi**2))
        u = np.zeros(N+1)
        u[1:N]= V[:,k]
        plt.plot(x,u,'x',color = 'r') 
        xx = np.linspace(-1,1,100)
        uu = np.polyval(np.polyfit(x.flatten(),u,N),xx)
        plt.plot(xx,uu,color ='k')
    plt.show()

elif program_number == 16: # Poisson eq. on [-1,1] x [-1,1] with u=0 on bndry
    N = 24 #number chebychev points
    (D,x) = cheb(N) # the differentiation matrix.
    xm,ym = np.meshgrid(x[1:N],x[1:N]) #actual grid.
    xm = np.ndarray.flatten(xm) # flatten the 2D grids into vectors
    ym= np.ndarray.flatten(ym)
    f= 10*np.sin(8*xm*(ym-1)) # forcing function
    D2 = (D.dot(D))[1:N,1:N] # second derivative matrix. 
    I =np.eye(N-1)
    L = np.kron(I,D2) + np.kron(D2,I) #two dimensional Laplacian matrix.
    u= solve(L,f) # solve the Poisson problem
    uu =np.zeros((N+1,N+1))
    uu[1:N,1:N]= u.reshape((N-1,N-1))
    x= x.reshape(len(x))
    xx,yy = np.meshgrid(x,x)
    xxx,yyy = np.meshgrid(np.linspace(-1,1,400),np.linspace(-1,1,400))
    f= interpolate.interp2d(xx,yy,uu)
    uuu = f(np.linspace(-1,1,400),np.linspace(-1,1,400))
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot_surface(xxx,yyy,uuu,cmap ='jet')
    ax.set_zlabel('u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('u(0,0)= {:e}'.format(uu[N/4,N/4]))
    plt.show()
elif program_number ==17: # Helmholtz eq. on [-1,1]x[-1,1] u_xx + u_yy +(k^2)u =f
    N = 24 # number of chebychev points
    (D,x) = cheb(N) # the differentiation matrix and chebychev grid.
    xm,ym = np.meshgrid(x[1:N],x[1:N])# mesh grid
    xm,ym = xm.flatten(),ym.flatten() # flatten 2D grids into vectors.
    f= np.exp(-10*((xm-1)**2 + (ym-0.5)**2))
    D2 = (D.dot(D))[1:N,1:N] #second derivative
    I = np.eye(N-1) # identity for the 2-D kronecker product.
    k = 9 # mode number for the Helmholtz problem
    L = np.kron(I,D2) + np.kron(D2,I) + (k**2)*np.eye((N-1)**2)
    u = solve(L,f) # solve the Helmholtz problem
    uu = np.zeros((N+1,N+1)) #to hold the solution 2D soln.
    uu[1:N,1:N] = u.reshape((N-1,N-1))
    xx,yy = np.meshgrid(x,x)
    xf,yf = np.linspace(-1,1,400),np.linspace(-1,1,400) # finer grid
    xx_f,yy_f= np.meshgrid(xf,yf) #mesh of the finer grid.
    f= interpolate.interp2d(xx,yy,uu)
    u_eval = f(xf,yf)
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(211, projection = '3d')
    ax.plot_surface(xx_f,yy_f,u_eval, cmap = 'PuOr')
    ax.set_zlabel('u')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('u(0,0) = {:e}'.format(uu[N/2,N/2]))
    ax2= fig.add_subplot(212)
    ax2.contour(xx_f,yy_f,u_eval)
    plt.show()

