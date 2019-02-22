# This code uses a precomputed mapping from a source plane
# to an observer plane to warp input from a webcam.
# I'm not sure how unviversal the webcam interface is, but it works on Ubuntu on my laptop.
# The lens is elliptical, a Gaussian peaked at the origin.
# The mapping was computed using the thin lens approximation.

# 'g' toggles a grid.
# 'c' toggles colour in each quadrant.
# 'l' toggles the lens.

# THINGS TO ADD
# Close window on clicking the x? At the moment it just reappears, only way to exit is through esc.
# Side-on view, showing actual geodesics.
# There isn't really any reason to use the thin lens approximation if you just precompute the geodesics.
# Move lens with mouse, or stacking multiple lenses on click.
# Make into some sort of game.
# Lens actual picture of galaxies.
# Make lens shrink and grow.

import numpy as np
import cv2

cv2.namedWindow("window")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

W = np.shape(frame)[0]
L = np.shape(frame)[1]

N = 480 # Size of lens is N*N
# Load in prev calc'd lens mapping.
# Let I = [-1,1].
# Lens mapping was IXI -> IXI,
# so need to rescale to [0,W]X[0,L] (pixels)
lens_mapping = np.load('lens_mapping_480.npy')
lens_mapping[:,:,0] = np.round((lens_mapping[:,:,0]+1)*0.5*W)
lens_mapping[:,:,1] = np.round((lens_mapping[:,:,1]+1)*0.5*L)
lens_mapping = lens_mapping.astype(int) # Discrete pixels
rescaled_mapping = np.array([[[int(j*N*1./W),int(i*N*1./L)] for i in range(L)] for j in range(W)])
rescaled_mapping = rescaled_mapping.reshape(W*L,2)
lens_mapping = lens_mapping.reshape(N*N,2)
rescaled_mapping = rescaled_mapping[:,0]+rescaled_mapping[:,1]*N
lens_mapping = lens_mapping[:,0]*L+lens_mapping[:,1]

inds = np.arange(W*L)
l_tenths = (((inds%L)%(L/10))<2) + (((inds%L)%(L/10))==(L/10)-1)
w_tenths = ((((inds-inds%L)/L)%(W/10))<2) + ((((inds-inds%L)/L)%(W/10))==(W/10)-1)

lgrid = (l_tenths!=0)
wgrid = (w_tenths!=0)

top = slice(0,W/2)
left = slice(0,L/2)
bottom = slice(W/2+1,-1)
right = slice(L/2+1,-1)

reindexing = lens_mapping[rescaled_mapping]
lensing = True
grid = False
colours = False
print 'Press ESC to quit'
while rval:
    frame = frame[:,::-1]           # Mirror the image
    if colours:
        frame[top,left] *= np.array([1,0,0],dtype='uint8')
        frame[top,right] *= np.array([0,1,0],dtype='uint8')
        frame[bottom,left] *= np.array([0,0,1],dtype='uint8')
        frame[bottom,right] *= np.array([1,0,1],dtype='uint8')
    frame = frame.reshape(W*L,3)    # 2d rgb array to 1d rgb array
    if grid:
        frame[lgrid] = np.array([1,1,1])
        frame[wgrid] = np.array([1,1,1])
    if lensing:
        frame = frame[reindexing]       # Apply lens
    frame = frame.reshape(W,L,3)    # Back to 2d rgb array

    cv2.imshow("window", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 108:
        lensing = not lensing
    elif key == 103:
        grid = not grid
    elif key == 99:
        colours = not colours
    elif key != -1:
        print key
        print 'Press ESC to quit'
    elif cv2.getWindowProperty('window', 0) != 0:
        print cv2.getWindowProperty('window', 0)

cv2.destroyWindow("window")
