import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sbs
sbs.set()
'''
# Orbit
x0 = 10.
y0 = 0
vx0 = 0
vy0 = 1./np.sqrt(x0)
'''
N = 200
x0 = 10.
y0s = np.array([0.5+np.sin(2*np.pi*i*1./N) for i in range(N)])
z0s = np.array([np.cos(2*np.pi*i*1./N) for i in range(N)])
#y0s = np.array([1 for i in range(N)])
#z0s = np.array([i*2.0/(N-1)-1. for i in range(N)])
vx0 = -6.
vy0 = 0.
vz0 = 0.

def derivs(y,t):
    photon = 1.
    dy      = np.zeros_like(y)
    dy[0:3] = y[3:6]
    r2      = np.sum(y[0:3]**2)
    dy[3:6] = -(photon+1.)*y[0:3]/r2**1.5
    return dy

ysi = []
zsi = []
ysf = []
zsf = []
for y0,z0 in zip(y0s,z0s):
    vals= np.array([x0,y0,z0,vx0,vy0,vz0])
    ts = np.linspace(0,1e1,1000)

    soln = odeint(derivs,vals,ts)
    xs = soln[:,0]
    ys = soln[:,1]
    zs = soln[:,2]
    target_index = np.where(xs<-x0)[0][0]
    ysi.append(ys[0])
    zsi.append(zs[0])
    ysf.append(ys[target_index])
    zsf.append(zs[target_index])
    vxs = soln[:,3]
    vys = soln[:,4]
    vzs = soln[:,5]

    max_dist = max(x0,y0,z0)
    #plt.xlim(-x0,x0)
    #plt.xlim(-y0*1.1,y0*1.1)
    #plt.ylim(0,y0*1.1)
    plt.subplot(2,2,1)
    #plt.xlim(-max_dist,max_dist)
    #plt.ylim(-max_dist,max_dist)
    plt.plot(xs[:target_index],ys[:target_index],'-')
    plt.subplot(2,2,3)
    plt.plot(xs[:target_index],zs[:target_index],'-')
plt.subplot(2,2,2)
plt.plot(zsi,ysi,'-')
plt.subplot(2,2,4)
plt.plot(zsf,ysf,'-')
plt.show()

'''
r2s = xs**2+ys**2
Vs  = -1./np.sqrt(r2s)
Ks  = 0.5*(vxs**2+vys**2)
E0 = Vs[0]+Ks[0]
plt.plot(ts,(Vs+Ks)/E0,'-')
plt.plot(ts,Vs/E0)
plt.plot(ts,Ks/E0)
plt.show()
'''



