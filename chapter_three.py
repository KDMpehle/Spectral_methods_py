#Python scripts for chapter three of Trefethen's 'spectral methods'
# in matlab', translated by Khaya Mpehle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.linalg import toeplitz
program_number = 4 # this number selects which program you wish to run.
if program_number == 4:
    #program 4: Period spectral differentiation
    N=24
    h= 2*np.pi/N
    xx = np.arange(h,2*np.pi +h,h).T #grid set up
    NN = np.array(range(1,N))
    cols = np.concatenate([[0], 0.5*np.power(-1,NN)/np.tan(NN*h/2)]).T #columns of Toeplitz matrix
    rows = np.concatenate([[cols[0]], cols[:0:-1]]) # row of Toeplitz matrix
    D= toeplitz(cols,rows)
    #differentiation of exp(sin(x))
    v= np.exp(np.sin(xx))
    vprime = np.cos(xx)*v
    fig = plt.figure(figsize=(16,8))
    ax = plt.axes()
    error = np.linalg.norm(D.dot(v)-vprime,np.inf)
    plt.subplot(2,2,1)
    plt.plot(xx,v,'--o')
    plt.title('Function',fontsize = 16)
    plt.subplot(2,2,2)
    plt.plot(xx,D.dot(v),'--o')
    plt.title("Spectral derivative",fontsize = 16)
    plt.text(2.1,1.,r'max error = %.5e'%error)
    # differentiation of a triangle function
    v = np.where(np.maximum(0,1-np.abs(xx-np.pi)/2),1,0)*(1-np.abs(xx-np.pi)/2)
    plt.subplot(2,2,3)
    plt.plot(xx,v,'--o')
    plt.subplot(2,2,4)
    plt.plot(xx,D.dot(v),'--o')
    plt.show()
elif program_number ==5:
    #program 5: repetition of program 4 via the Fast Fourier Transform.
    N=24
    h= 2*np.pi/N
    xx= np.arange(h,2*np.pi+h,h).T # define the real space grid
    # the function exp(sin(x)) to be found
    v= np.exp(np.sin(xx))
    vprime = np.cos(xx)*v
    vhat = np.fft.fft(v)
    coeffs= [np.array(range(0,int(N/2))),[0],np.array(range(-int(N/2)+1,0))]
    w_hat = 1j*np.concatenate(coeffs).T*vhat
    W = np.real(np.fft.ifft(w_hat))
    error = np.linalg.norm(W-vprime,np.inf)
    plt.subplot(2,2,1)
    plt.plot(xx,v,'--o')
    plt.title('Function')
    plt.subplot(2,2,2)
    plt.plot(xx,W, '--o')
    plt.title('FFT derivative')
    plt.text(2.1,1.,r'max error = %.5e'%error)
    print('maximum inf norm error is', error)
    #now differentiate the hat function
    v= np.where(np.maximum(0,1-np.abs(xx-np.pi)/2),1,0)*(1-np.abs(xx-np.pi)/2)
    vhat = np.fft.fft(v)
    w_hat = 1j*np.concatenate(coeffs).T*vhat
    W = np.real(np.fft.ifft(w_hat))
    plt.subplot(2,2,3)
    plt.plot(xx,v,'--o')
    plt.subplot(2,2,4)
    plt.plot(xx,W, '--o')
    plt.show()

elif program_number == 6:
    #program 6: the variable coefficient transport equation
    N = 128
    h = 2*np.pi/N # uniform mesh size.
    xx = np.arange(h, 2*np.pi+h,h) # spatial grid
    t,dt,T = (0,h/4,8) # time step and time horizon.
    def wave_speed(x): # variable wave speed
        return 0.2 +np.sin(x-1)**2
    def v0(x): #initial condition
        return np.exp(-100*(x-1)**2)
    def vold(x):#reverse step for the leap-frog method.
        return np.exp(-100*(x - 0.2*dt -1)**2)
    c= wave_speed(xx)
    v=v0(xx)
    v_minus = vold(xx)
    tplot = 0.15
    plotgap= int(tplot/dt) #gap between plots
    dt = tplot/plotgap #adjust time step.
    nplots = int(T/tplot) # number of plots
    data =np.zeros((nplots+1,N))
    data[0,:] = v
    tdata = [t]
    for i in range(0,nplots):
        for n in range(0,plotgap):
            t = t+dt # march forward in time
            vhat = np.fft.fft(v) # FFT of the solution
            coeffs= [np.array(range(0,int(N/2))),[0],np.array(range(-int(N/2)+1,0))]
            w_hat = 1j*np.concatenate(coeffs)*vhat# spectral differentiation.
            w= np.real(np.fft.ifft(w_hat))
            vnew = v_minus -2*dt*c*w # leap frog method.
            v_minus = v # update old solution to be current one
            v=vnew # update current solution as vnew
        data[i+1,:] = v
        tdata = np.concatenate([tdata,[t]])
    #Rough python equivalent of waterfall plot
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    xm,tm = np.meshgrid(xx,tdata) # meshgrid of space and time
    ax.plot_wireframe(xm,tm,data,cstride = 100)
    ax.set_ylabel('t')
    ax.set_xlabel('x')
    ax.set_zlabel('u')
    ax.set_xlim3d([0,2*np.pi])
    ax.set_ylim3d([0,5])
    ax.set_zlim3d([0,4])
    ax.set_title(r'solution of $u_t = -c(x)u_x$, $c(x)= \frac{1}{5} + \sin^2(x-1)$')
    plt.show()
