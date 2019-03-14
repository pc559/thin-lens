import numpy as np
import imageio
import sys

filename = sys.argv[1]
frame = imageio.imread(filename)
if np.shape(frame)[2]>3:
    frame = frame[:,:,:3]
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

frame = frame.reshape(W*L,3)    # To 1d rgb array
frame = frame[reindexing]       # Apply lens
frame = frame.reshape(W,L,3)    # Back to 2d rgb array

output_name = filename.split('/')[-1][:-4]+'_lensed.png'
imageio.imwrite(output_name, frame)



