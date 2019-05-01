#Python scripts for chapter five of Trefethen's 'spectral methods
# in matlab', translated by Khaya Mpehle
import numpy as np
import matplotlib.pyplot as plt
from chapter_six import cheb
program_number = 10 # this number selects which program you wish to run.
if program_number == 9:
    N = 16
    xx = np.linspace(-1.01,1.01,200)
    for i in range(2):
        if i ==0:
            text = 'equispaced points'
            x = -1 +2*np.array(range(0,N+1))/N
        elif i == 1:
            text = 'Chebycheff points'
            x= np.cos(np.pi*np.array(range(0,N+1))/N)
        plt.subplot(2,1,i)
        u = 1./(1 + 16*x**2)
        uu = 1./(1 + 16*xx**2)
        pp = np.polyval(np.polyfit(x,u,N),xx)
        plt.plot(x,u,'o', ms = 8, color = 'blue')
        plt.plot(xx,pp,color = 'blue')
        plt.axis([-1.1,1.1,-1,1.5])
        plt.title(text)
        error = np.linalg.norm(uu -pp, np.inf)
        plt.text(-0.5,-0.5,'max error = {:f}'.format(error))
    plt.show()

elif program_number == 10:# polynomial and their equipotential curves.
    N = 16
    for i in range(2):
        if i == 0:
            text= 'equispaced points'
            x= -1 +2*np.array(range(0,N+1))/N
        elif i == 1:
            text = 'Chebychev points'
            x = np.cos(np.pi*np.array(range(0,N+1))/N)
        p = np.poly(x)
        xx = np.linspace(-1,1,200)
        pp = np.polyval(p,xx)# plot p(x) on standard interval.
        plt.subplot(2,2,2*i-1)
        plt.plot(x,np.zeros(len(x)),'o', ms = 8)
        plt.plot(xx,pp)
        plt.title(text)
        plt.subplot(2,2,2*i) # Now plot the equipotential curves.
        plt.plot(np.real(x),np.imag(x),'o',ms = 8)
        plt.axis([-1.4,1.4,-1.12,1.12])
        xm,ym = np.linspace(-1.4,1.4,50), np.linspace(-1.12,1.12,50)
        xx, yy = np.meshgrid(xm,ym) # meshgrid of points.
        zz = xx + 1j*yy
        pp = np.polyval(p,zz)
        levels = [10**m for m in range(-4,1)]
        plt.contour(xx,yy,np.absolute(pp),levels)
        plt.title(text)
    plt.show()
    
