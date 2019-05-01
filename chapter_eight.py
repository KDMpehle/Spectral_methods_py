# Python scripts for chapter 8 of L.N. Trefethen's 'Spectral methods'
# in matlab, translated by Khaya Mpehle
#chapter eight: Chebycheff series and the FFT
import numpy as np
import matplotlib.pyplot as plt
from chapter_six import cheb
from mpl_toolkits.mplot3d import axes3d
program_number = 19

def chebfft(v):
    '''
    Enacts Chebycheff differentiation via the fft, note it could be
    more efficient if a discrete cosine transform were used.
    '''
    N = len(v) - 1
    if N==0:
        w = 0
        return w
    NN = np.array(range(0,N+1))# array of grid point indices
    x=np.cos(np.pi*NN/N)# reshape into column vector.
    inds = np.array(range(0,N))
    v= v.flatten() # flatten the input vector.
    V= np.concatenate((v,np.flipud(v[1:N])))
    U = np.real(np.fft.fft(V))
    AA = 1j*np.concatenate([inds,[0],np.array(range(1-N,0))])
    W = np.real(np.fft.ifft(AA*U))
    w= np.zeros(N+1)
    w[1:N] = -W[1:N]/np.sqrt(1-x[1:N]**2)
    NN = np.array(range(1,N+1))
    w[0]= np.sum(inds**2*U[inds])/N + 0.5*N*U[N]
    w[N] = ( np.sum(np.power(-1,NN)*inds**2*U[inds])/N
             +0.5*(-1)**(N+1)*N*U[N] )
    return w

if program_number == 18:
    xx = np.linspace(-1,1,100)
    ff = np.exp(xx)*np.sin(5*xx)
    for N in [10, 20]:
        x= np.cos(np.pi*np.array(range(0,N+1))/N) # relevant Chebycheff points.
        f= np.exp(x)*np.sin(5*x) # function on Chebychev points.
        plt.subplot(2,2,2*(N/10)-1)
        plt.plot(x,f,'x', ms =6,label = 'f(x)')
        plt.plot(xx,ff,'-',label = 'interpolated')
        plt.legend(loc= 'lower left')
        plt.title('f(x), N = {:d}'.format(N))
        error = chebfft(f) -np.exp(x)*(np.sin(5*x) +5*np.cos(5*x))
        plt.subplot(2,2,2*(N/10))
        plt.plot(x,error,'-o',ms=6)
        plt.title(" error in f'(x), N = {:d}".format(N))
    plt.show()

elif program_number == 19: # 2nd-order wave eq. on Chebychev grid (p6.m)
    N= 80 # number of Chebycheff points.
    x,dt = np.cos(np.pi*np.array(range(0,N+1))/N),8./(N**2) #grid and time step.
    v, vold = np.exp(-200*x**2), np.exp(-200*(x-dt)**2)
    t,tmax, tplot= (0,4, 0.075)
    plotgap, nplots = int(tplot/dt), int(tmax/tplot)
    plotgap, dt= int(tplot/dt), tplot/plotgap
    data = np.zeros((nplots +1, N+1)) # solution at each time step matrix.
    data[0,:] = v
    tdata = [t]
    for i in range(0,nplots):
        for n in range(0,plotgap):
            t=t+dt
            w = chebfft(chebfft(v))
            w[0], w[N] = 0,0
            vnew = 2*v -vold +dt**2*w
            vold = v
            v = vnew
        data[i+1,:] = v
        tdata = np.concatenate([tdata,[t]])
    #Rough Python equivalent of waterfall plot.
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    xm,tm = np.meshgrid(x,tdata) # meshgrid of space and time.
    ax.plot_wireframe(xm,tm,data,cstride =200)
    ax.set_ylabel('t')
    ax.set_xlabel('x')
    ax.set_zlabel('u')
    ax.set_xlim3d([-1,1])
    ax.set_ylim3d([0,tmax])
    ax.set_zlim3d([-2,2])
    ax.set_title(r'Solution of equation $u_{tt} = u_{xx}$, u(1)=u(-1) =0')
    plt.show()
            
    
