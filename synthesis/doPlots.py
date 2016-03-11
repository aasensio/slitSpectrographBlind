import numpy as np
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
import scipy.io as scio

filename = '/net/nas4/fis/aasensio/3dcubes/jaime/original_and_degraded.nf'
f = scio.netcdf_file(filename, 'r')

pl.close('all')
fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10,12))
ax[0,0].imshow(f.variables['degraded'][0,:,:,0], cmap='viridis')
ax[1,0].imshow(f.variables['degraded'][0,:,:,10], cmap='viridis')
ax[2,0].imshow(f.variables['original'][0,:,:], cmap='viridis')
ax[0,1].imshow(f.variables['degraded'][25,:,:,0], cmap='viridis')
ax[1,1].imshow(f.variables['degraded'][25,:,:,10], cmap='viridis')
ax[2,1].imshow(f.variables['original'][25,:,:], cmap='viridis')
ax[0,0].set_title('Degraded continuum')
ax[0,1].set_title('Degraded line core')
ax[1,0].set_title('Degraded continuum')
ax[1,1].set_title('Degraded line core')
ax[2,0].set_title('Original continuum')
ax[2,1].set_title('Original line core')