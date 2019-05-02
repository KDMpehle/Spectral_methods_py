#Python scripts for chapter thirteen of Trefethen's 'spectral methods
# in matlab', translated by Khaya Mpehle.
#chapter thirteen: More about boundary conditions
import numpy as np
import matplotlib.pyplot as plt
from chapter_six import cheb
from scipy.linalg import solve, toeplitz
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

program_number = 36

if program_number == 32:# u_xx = exp(4x), -1<x<1 u(-1) =0, u(1)=1
    N=16 # Chebychev points: use Chebychev collocation
    (D,x) = cheb(N) # Chebychev differentiation matrix
    print(x)
    D2=D.dot(D) #second derivative matrix
    D2 = D2[1:N,1:N]
    f= np.exp(4*x[1:N])
    u = solve(D2,f) # solve for the u vector
    u= np.vstack([[0],u,[0]]) + (x+1)/2 # the exact solution
    xm = np.arange(-1,1+0.01,0.01) # a finer grid.
    x= x.flatten()
    u=u.flatten()
    print(x)
    uu= np.polyval(np.polyfit(x,u,N),xm)
    uext = (np.exp(4*xm) -np.sinh(4)*xm -np.cosh(4))/16 + (xm +1)/2
    fig = plt.figure(figsize = (16,8))
    ax = plt.axes()
    ax.plot(xm,uu,'x')
    ax.plot(xm,uext)
    ax.set_title('max error = %s'%(np.linalg.norm(uu-uext,np.inf)))
    plt.show()
elif program_number == 33:#solve u_xx = exp(4x), u'(-1)= u(1)=0
    N = 16 # number of grid points
    (D,x) = cheb(N) # chebychev grid and differentiation matrix.
    D2 = D.dot(D) #second derivative matrix.
    D2[N,:]  = D[N,:] # extra Neuman equation.
    D2=D2[1:N+1,1:N+1]
    f = np.exp(4*x[1:N])
    u = solve(D2,np.vstack([f,[0]]))
    u = np.vstack([[0],u]) # add u(1) = 0 on the boundary.
    xm = np.arange(-1,1+0.01,0.01) # finer grid
    x,u= x.flatten(), u.flatten() # flatten to 'row vectors'
    uu = np.polyval(np.polyfit(x,u,N),xm)
    uext = (np.exp(4*xm) -4*np.exp(-4)*(xm-1) -np.exp(4))/16 # exact solution.
    ax = plt.axes()
    ax.plot(xm,uu,'x')
    ax.plot(xm,uext)
    ax.set_title('max error = %s'%(np.linalg.norm(uext-uu,np.inf)))
    plt.show()

elif program_number == 36:# Laplace equation on [-1,1]x[-1,1] nonzero bcs
    N = 24 # number of Chebychev grids in each direction.
    (D,x) = cheb(N) # differentiation matrix and 1-D lattices.
    xm,ym = np.meshgrid(x,x)
    xm,ym = xm.flatten(),ym.flatten() # flatten 2-D mesh to 1-d vectors.
    print(xm)
    D2, I = D.dot(D), np.eye(N+1) # second derivative and identity.
    L = np.kron(I,D2) + np.kron(D2,I) # 2-D Laplacian.
    #impose boundary conditions
    b = np.nonzero(((np.abs(ym) ==1)|(np.abs(xm)==1)))[0]#indices corresponding to boundary.
    b=b.tolist()
    print(len(b))
    print(len(b),4*N)
    print(b)
    L[b,:] = np.zeros((4*N,(N+1)**2))
    print(L.shape,L[np.ix_(b,b)].shape)
    L[np.ix_(b,b)] = np.eye(4*N)
    rhs = np.zeros((N+1)**2)
    rhs[b] =( ((ym[b]==1).astype(np.int))*((xm[b]<0).astype(np.int))*np.sin(np.pi*xm[b])**4 
              +0.2*np.array(((xm[b]==1).astype(np.int)))*np.sin(3*np.pi*ym[b]) )
    u = solve(L,rhs) # solve the linear system
    uu = u.reshape((N+1,N+1))
    xm, ym = np.meshgrid(x,x)
    xf, yf = np.linspace(-1,1,100),np.linspace(-1,1,100) #finer x,y arrays.
    xm_f,ym_f = np.meshgrid(xf,yf) # meshgrid of finer points.
    f= interpolate.interp2d(xm,ym,uu,kind = 'cubic')
    uuu = f(xf,yf)
    print(uu[N/2,N/2])
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot_surface(xm_f,ym_f,uuu)
    plt.show()

elif program_number == 37: #Wave equation with periodic and Neumann conditions
    #x variable on [-A,A] discretised as Fourier collocation
    A = 3 # half length of periodic domain.
    Nx = 50 # number of Fourier points
    dx = 2*A/Nx
    x = np.arange(-A +dx, A + dx,dx) # Fourier grid points
    MM= np.array(range(0,M)) + 1
    Rowcol = np.concatenate([[-np.pi**2/((3*dx/A)**2) - 1./6],
                            0.5*np.power(-1,MM[1:M])/(np.sin(dx*(MM[0:M-1]/2))**2)])
    D2x = toeplitz(Rowcol)
    # y variable in [-1,1] with Chebysheff coordinates.
    Ny = 15 # number of Chebysheff points.
    (Dy,y) = cheb(Ny)
    D2y = Dy.dot(Dy) # second derivative operator.
    BC = solve(-Dy[np.ix_([0,Ny],[0,Ny])], Dy[(0,Ny),1:Ny-1])
    # initial data.
    xm,ym = np.meshgrid(x,y)
    vv = np.exp(-8*((xx + 1.5)**2 +yy**2))
    vvold = np.exp(-8*((xx +dt + 1.5)**2 + y**2))
    #time-stepping via the leapfrog procedure
    dt = 5./(Nx + Ny**2)
    plotgap = round(2/dt)
    dt = 2./plotgap
    for n in range(2*plotgap + 1):
        t = n*dt
        ###fill the rest
    vvnew = 2*vv -vvold +dt**2*(vv*D2x + D2y*vv)
    vvold = vv # backwards
    vv= vvnew # set previous new solution to be the mid solution
    #Apply the Neumann boundary condition.
    
    
