import numpy as np
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
import astropy.io.fits as fits
import scipy.io as scio
import pyiacsun as ps
from numba import jit

def hanningWindow(nPix, percentage):
    """
    Return a Hanning window in 2D

    Args:
    size (int): size of the final image
    percentage (TYPE): percentage of the image that is apodized

    Returns:
    real: 2D apodization mask

    """ 
    M = np.ceil(nPix*percentage/100.0)
    win = np.hanning(M)

    winOut = np.ones(nPix)
    winOut[0:M/2] = win[0:M/2]
    winOut[-M/2:] = win[-M/2:]

    return np.outer(winOut, winOut)

# @jit
# def conv(spec, psf, nPixBorder):    
#     nx, ny, nlambda = spec.shape
#     nxPSF, nyPSF, nPSF = psf.shape
#     out = np.zeros_like(spec)
#     for i in range(nx-2*nPixBorder):
#         for j in range(ny-2*nPixBorder):
#             for k in range(nxPSF):
#                 for l in range(nyPSF):                    
#                     out[i,j,0] += spec[i+k-nxPSF/2+nPixBorder,j+l-nyPSF/2+nPixBorder,0] * psf[k,l,i]
#     return out


print("Reading data...")
hdu = fits.open('/net/nas4/fis/aasensio/3dcubes/jaime/spectra3d.t232.5_254x254x82_mu=1.0_vp.fits')
spec = hdu[0].data

hdu = fits.open('/net/nas4/fis/aasensio/3dcubes/jaime/wav.t232.5_254x254x82_mu=1.0_vp.fits')
wave = hdu[0].data

f = scio.netcdf_file('/net/nas4/fis/aasensio/3dcubes/jaime/psfs.nf', 'r')
psf = f.variables['psf'][:]

nPixPSF, nPixPSF, nStepsPSF, nImagesPSF = psf.shape

nPixBorder = int(nPixPSF/2)

lambdaStart = 1355
lambdaEnd = 1445
wave = wave[lambdaStart:lambdaEnd]
spec = spec[0:128+nPixPSF,0:128+nPixPSF,lambdaStart:lambdaEnd].T

nLambda, ny, nx = spec.shape

left = nPixBorder
right = nx - nPixBorder
top = nPixBorder
bottom = ny - nPixBorder

res = np.zeros((nLambda,128,128))

f = scio.netcdf_file('/net/nas4/fis/aasensio/3dcubes/jaime/original_and_degraded.nf', 'w')
f.createDimension('nPix', 128)
f.createDimension('nImages', nImagesPSF)
f.createDimension('nLambda', nLambda)
varOriginal = f.createVariable('original', 'd', ('nLambda', 'nPix', 'nPix', 'nImages'))
varDegraded = f.createVariable('original', 'd', ('nLambda', 'nPix', 'nPix', 'nImages'))

for k in range(nImagesPSF):
    for i in range(left,right):
        ps.util.progressbar(i-left, right-left)
        currentPSF = psf[:,:,i-left,k] / np.sum(psf[:,:,i-left,k])
        for j in range(top,bottom):
            res[:,i-left,j-top] = np.sum(spec[:,i-nPixBorder:i+nPixBorder,j-nPixBorder:j+nPixBorder] * currentPSF[None,:,:], axis=(1,2))

    varOriginal[:,:,:,k] = spec[:,left:right,top:bottom]
    varDegraded[:,:,:,k] = res
        # out = conv(spec, psf[:,:,:,0], nPixBorder)

f.close()


pl.close('all')
f, ax = pl.subplots(nrows=2, ncols=2, figsize=(10,10))
ax[0,0].imshow(res[0,:,:])
ax[1,0].imshow(spec[0,left:right,top:bottom])
ax[0,1].imshow(res[25,:,:])
ax[1,1].imshow(spec[25,left:right,top:bottom])