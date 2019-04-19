# Python scripts for chapter 14 of Trefethen's 'spectral methods'
# in matlab, translated by Khaya Mpehle
#chapter fourteen: Fourth order problems.
import numpy as np
import matplotlib.pyplot as plt
from chapter_six import cheb
from scipy.linalg import solve,eig
from numpy.linalg import norm
from scipy import interpolate
program_number = 40 #select which program to run

if program_number == 38: # solve u_IV = exp(x) with clamped BC on [-1,1]
    N=15 #Chebychev points
    (D,x) = cheb(N) # Chebychev differentiation matrix.
    S = np.diag(np.insert(1./(1-x[1:N]**2),[0,len(x[1:N])],[0,0]))
    D2 = np.dot(D,D)
    D3,D4 = np.dot(D,D2),np.dot(D2,D2)
    xf = x.flatten()
    D4 = ((np.diag(1-xf**2)).dot(D4) - 8*(np.diag(xf)).dot(D3)-12*D2).dot(S)
    D4 = D4[1:N,1:N] # apply the clamed boundary conditions.
    #Move on to solve the boundary value problem.
    f= np.exp(x[1:N])
    u= np.zeros(N+1)
    u[1:N]= solve(D4,f).flatten()
    plt.plot(x,u,'x',color='red')
    xx= np.linspace(-1,1,100)
    uu = (1-xx**2)*np.polyval(np.polyfit(x.flatten(),S.dot(u),N),xx)
    plt.plot(xx,uu,'k')
    plt.show()
    #determine the exact solution and compare the error.
    A = np.array([[1,-1,1,-1],[0,1,-2,3],[1,1,1,1],[0,1,2,3]])
    V= np.vander(xx)
    V= V[:,:-5:-1] #last four columns of the vandermonde matrix.
    c= solve(A,np.exp(np.array([-1,-1,1,1])))
    uext = np.exp(xx) - V.dot(c) # exact solution
    print(norm(uu- uext,np.inf)) # accuracy of 1E-15 for N = 15 points.

elif program_number == 39: #Eigenvalues of the biharmonic operator on square with clamped BCs
    #something wrong: correct eigenvalues but eigenvectors odd...
    N =17
    (D,x) = cheb(N) # get the differentiation matrix.
    xf= x.flatten()
    D2 = D.dot(D)
    D3, D4 = np.dot(D,D2), np.dot(D2,D2)
    S = np.diag(np.insert(1./(1-x[1:N]**2),[0,len(x[1:N])],[0,0]))
    D4 = ((np.diag(1-xf**2)).dot(D4) -8*(np.diag(xf)).dot(D3) -12*D2).dot(S) # 1-D matrix
    I = np.eye(N-1) #identity.
    D2,D4 = D2[1:N,1:N], D4[1:N,1:N]
    L = np.kron(I,D4) + np.kron(D4,I) +2*np.kron(D2,I).dot(np.kron(I,D2))
    #25 eigenvalues and associated eigenfunctions.
    W, V = np.linalg.eig(-L)
    lamb = -np.real(W)
    V= np.real(V)
    inds = np.argsort(lamb)
    Evals = lamb[inds]
    V=V[:,inds]
    print(np.shape(V))
    Evals = np.sqrt(Evals/Evals[0])
    xx,yy = np.meshgrid(x[1:N],x[1:N])
    xx = np.reshape(xx,(N-1,N-1))
    yy= np.reshape(yy,(N-1,N-1))
    xf,yf = np.linspace(-1,1,100),np.linspace(-1,1,100)
    xxx,yyy = np.meshgrid(xf,yf)
    plt.figure(figsize = (10,10))
    for i in range(0,10):
        print(Evals[i])
        plt.subplot(5,2,i+1)
        uu= np.zeros((N+1,N+1))
        print(np.shape(V[:,i]))
        uu[1:N,1:N] =np.reshape(V[:,i],(N-1,N-1))
        uu = uu/norm(uu,np.inf)
        f= interpolate.interp2d(x,x,uu,kind = 'cubic')
        uuu = f(xf,yf)
        plt.contour(xf,yf,uuu)
        plt.colorbar()
        plt.title(r'$\lambda$ = %18.2f'%(Evals[i]))
    plt.show()

elif program_number == 40:
    #Eigenvalues of the Orr-Sommerfeld operator
    Re = 5772# the Reynolds number.
    for N in range(40,120,20):
        (D,x) = cheb(N) #number of Chebychev points (variable here).
        S = np.diag(np.insert(1./(1-x[1:N]**2),[0,len(x[1:N])],[0,0]))
        D2 = np.dot(D,D)
        D3,D4 = np.dot(D,D2),np.dot(D2,D2)
        xf = x.flatten()
        D4 = ((np.diag(1-xf**2)).dot(D4) - 8*(np.diag(xf)).dot(D3)-12*D2).dot(S)
        D2,D4=D2[1:N,1:N], D4[1:N,1:N] # spectral fourth derivative matrix.
        #Generalised eigenvalue probem.
        I = np.eye(N-1)
        A = (D4 -2.*D2+I)/Re -2j*I-1j*np.dot(np.diag(1-xf[1:N]**2),D2-I)
        B = D2 - I
        W,V = eig(A,B)
        plt.subplot(4,2,N/20-1)
        plt.xlim([-1,0.2])
        plt.ylim([-1,0])
        plt.scatter(np.real(W),np.imag(W))
        print(np.max(np.real(W)))
        #plt.title('max lamb = %15.11f"'%(np.max(np.real(W))))
    
    plt.show()
    
