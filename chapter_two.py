# Python scripts for chapter two of Trefethen's 'spectral methods'
# in matlab, translated by Khaya Mpehle
import numpy as np
import matplotlib.pyplot as plt

#program 3: The band-limited interpolation (BLI)
h, xmax = (1.,10.)
x = np.arange(-xmax,xmax+h,h)#x-grid
xx= np.arange(-xmax-h/20,xmax+h/20+h/10,h/10)#Grid for plotting
fig = plt.figure(figsize =(24,8))
for plot_range in range(3): #cases for various functions to interpolate.
    if plot_range == 0:
        v= np.where(abs(x) < 1e-6,1,0) # "impulse at origin.
    elif plot_range == 1:
        v= np.where(abs(x) <=3, 1, 0) # Hat function
    elif plot_range == 2:
        v= np.where(np.maximum(0,1.-np.abs(x)/3),1,0)*(1-np.abs(x)/3) #triangle function
    plt.subplot(4,1,plot_range+1)
    plt.plot(x,v)# plot the triangle function
    p =np.zeros(len(xx))#array of zeros to add terms of BLI
    for i in range(0,len(x)):
        p =p+ v[i]*np.sin(np.pi*(xx-x[i])/h)/(np.pi*(xx-x[i])/h)# terms of BLI
    plt.plot(xx,p,'x',color='k')
    plt.ylim([-0.3,1.2])
plt.show()
