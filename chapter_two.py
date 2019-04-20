# Python scripts for chapter two of Trefethen's 'spectral methods'
# in matlab, translated by Khaya Mpehle
import numpy as np
import matplotlib.pyplot as plt
#program 3: The band-limited interpolation (BLI)
h, xmax = (1.,10.)
x = np.arange(-xmax,xmax+h,h)#x-grid
xx= np.arange(-xmax-h/20,xmax+h/20+h/10,h/10)#Grid for plotting
v = np.where(abs(x) < 3,1,0) # The hat function
fig = plt.figure(figsize =(24,8))
ax =plt.axes()
plt.plot(x,v)# plot the triangle function
p =np.zeros(len(xx))#array of zeros to add terms of BLI
for i in range(0,len(x)):
    p =p+ v[i]*np.sin(np.pi*(xx-x[i])/h)/(np.pi*(xx-x[i])/h)# terms of BLI
print(p)
plt.plot(xx,p,'x',color='k')    
plt.show()
