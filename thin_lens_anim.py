import sys
import numpy as np
from time import time
import math
from scipy.integrate import odeint,dblquad
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if len(sys.argv)>1 and sys.argv[1]=='bkgd':
    bkgd=True   # Background quadrants coloured differently.
else:
    bkgd=False

# Everything happens on [-1,1]X[-1,1]
def np_surface_density(x):  # Lens mass distribution
    beta = 10.
    a = 5.
    b = 2.
    return 3*np.exp(-beta*(a*x[:,0]**2+b*x[:,1]**2))

def np_col_func(t,coords):  # Source colour distr.
    beta = 600.
    xs = coords[:,:,1]
    ys = coords[:,:,0]
    a = 18.
    b = 2.
    mult = 1.
    rad = np.exp(-beta*(a*(xs+t%2-1)**2+b*ys**2))   # Einstein cross/ring
    rad += mult*np.exp(-beta*(a*(xs+t%2-1)**2+b*(ys+0.5)**2))
    rad += mult*np.exp(-beta*(a*(xs+t%2-1)**2+b*(ys-0.5)**2))
    g = rad
    if bkgd==True:  # Colour the quadrants different colours to see how they're lensed.
        r = 1.*(xs>0)
        b = 1.*(ys>0)
    else:           # Black background, white source.
        r = g
        b = g
    return np.array([r,g,b]).T

def deflect(coords,integr_coords,ps,Ns,integr_surface):
    x,y = coords[0],coords[1]
    integr_denom = (0.001+np.sum((coords-integr_coords)**2,axis=-1))*np.pi
    integrand_x = integr_surface*(x-integr_coords[:,0])/integr_denom
    integrand_y = integr_surface*(y-integr_coords[:,1])/integr_denom
    ax = 4*np.sum(integrand_x)/Ns**2
    ay = 4*np.sum(integrand_y)/Ns**2
    # Use the below to check results, though much slower.
    #ax = dblquad(lambda u, v: surface_density(u,v)*(x-u)/(0.001+(x-u)**2+(y-v)**2)/np.pi, -1, 1, -1, 1, epsrel=1e-2)[0]
    #ay = dblquad(lambda u, v: surface_density(u,v)*(y-v)/(0.001+(x-u)**2+(y-v)**2)/np.pi, -1, 1, -1, 1, epsrel=1e-2)[0]
    return x-ax,y-ay

def defl_coords(coords,integr_coords,ps,Ns,N,integr_surface):
    defl_coords_array = [[deflect(coords[j][i],integr_coords,ps,Ns,integr_surface) for i in range(N)] for j in range(N)]
    return np.array(defl_coords_array)

def updatefig(*args):   # Evolve the source distr.
    defl_coords_array,N = args[1],args[2]
    global t
    t+=0.02
    res_distr = np_col_func(t,defl_coords_array)
    im.set_array(res_distr)
    return im,

def plot_2d_distr(coords,cols,pos,title=''):
    plt.subplot(pos)
    plt.title(title)
    plt.imshow(cols,extent=(-1,1,1,-1),interpolation='bicubic')

N = 150 # Number of pixels, later interpolated
Ns = 101    # Lens is sampled at Ns^2 points

#### Coordinates for [-1,1]X[-1,1] ########################################################
coords = np.array([[[-1+i*2./(N-1),-1+j*2./(N-1)] for i in range(N)] for j in range(N)])
###########################################################################################

#### Source colour array at t=1 with no lens ##############################################
source_colours = np_col_func(1,coords)
###########################################################################################

#### Get lens colour array ################################################################
lens_distr = [[np_surface_density(np.array([coords[i][j]]))[0] for i in range(N)] for j in range(N)]
# r=g=b
lens_cols = np.array([[[lens_distr[i][j]]*3 for i in range(N)] for j in range(N)])
lens_cols = lens_cols/lens_cols.max()
###########################################################################################

#### Compute and store point-in-sky-indep. part of the integral ###########################
t1 = time()
ps = np.linspace(-1,1,Ns)
integr_coords = np.array([ [ps[i],ps[j]] for j in range(Ns) for i in range(Ns) ])
integr_surface = np_surface_density(integr_coords)
t2 = time()
print t2-t1
###########################################################################################

#### Compute thin lens integral for each point in sky #####################################
t3 = time()
defl_coords_array = defl_coords(coords,integr_coords,ps,Ns,N,integr_surface)
t4 = time()
t=0
res_distr = np_col_func(t,defl_coords_array)
t5 = time()
print t4-t3
print t5-t4
###########################################################################################

#### Plotting #############################################################################
# The source at t=1 with no lens
plot_2d_distr(coords,source_colours,121,'Source, no lens')
# The lens (scaling arbitrary, whiteness prop. to mass density)
plot_2d_distr(coords,lens_cols,122,'Lens')
plt.subplots_adjust(wspace=0.3)
fig = plt.figure()
im = plt.imshow(res_distr,animated=True,interpolation='bicubic',extent=(-1,1,1,-1))
# Save lens for use with webcam, or whatever.
#np.save('lens_mapping_'+str(N)+'.npy',defl_coords_array)
interval = round(300/Ns)
ani = animation.FuncAnimation(fig,updatefig,interval=interval,blit=True,fargs=[defl_coords_array,N])
plt.show()
###########################################################################################

