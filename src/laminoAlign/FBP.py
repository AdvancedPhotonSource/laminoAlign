import numpy as np
import laminoAlign as lam
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy


def FBP(sinogram, weights, cfg, vectors, geometries={}, astraConfig={}, timerOn=False):

    # calculate the original angles
    theta = np.pi - np.arctan2(vectors[:, 1], -vectors[:, 0])
    laminoAngle = np.pi/2 - np.arctan2(vectors[:, 2], vectors[:, 0]/np.cos(theta))

    # determine weights in case of irregular fourier space sampling
    theta = np.mod(theta - theta[0], np.pi)
    sortIdx = np.argsort(theta)
    thetaSort = theta[sortIdx]
    Nproj = len(sinogram)

    weights = np.zeros(Nproj)
    weights[1:Nproj-1] = -thetaSort[0:Nproj-2]/2 + thetaSort[2:]/2
    weights[0] = thetaSort[1] - thetaSort[0]
    weights[-1] = thetaSort[-1] - thetaSort[-2]
    weights[sortIdx] = weights*1  # unsort
    if np.any(weights > 2*np.median(weights)):
        weights[weights > 2*np.median(weights)] = np.median(weights)
    weights = weights/np.mean(weights)
    weights = weights*(np.pi/2/Nproj)*np.sin(laminoAngle)
    weights = weights.astype(np.float32)

    H = designFilter(sinogram.shape[2])

    # account for laminography tilt + unequal spacing of the tomo 
    # angles
    H = H[:, np.newaxis]*weights

    t0 = lam.utils.timerStart()
    lam.utils.freeAllBlocks()
    sinogram = applyFilter_chunked(sinogram, H, sinogram.shape[2], n=50)
    lam.utils.timerEnd(t0, "FBP: Apply Filter", timerOn)

    # Use astra toolbox to reconstruct the 3D image
    t0 = lam.utils.timerStart()
    rec, astraConfig, geometries = lam.astra.astraReconstruct(
        sinogram, cfg, vectors, geometries, astraConfig, timerOn)
    lam.utils.timerEnd(t0, "FBP: Astra Reconstruct", timerOn)

    return rec, astraConfig, geometries


def designFilter(width, d=1):

    order = np.max([64, 2**(np.ceil(np.log2(2*width)))])
    filt = np.linspace(0, 1, int(order/2), dtype=np.float32)
    # Frequency axis up to Nyquist
    w = np.linspace(0, np.pi, int(order/2), dtype=np.float32)
    # Crop the frequency response
    filt[w > np.pi*d] = 0 
    # Make filter symmetric
    filt = np.append(filt, filt[::-1]) 

    return filt


def applyFilter_chunked(sinogram, H, Nw, n, timerOn=False):

    Nangles = len(sinogram)
    Niter = int(np.ceil(Nangles/n))
    filteredSinogram = np.zeros(sinogram.shape, dtype=np.float32)
    m = H.shape[0]
    padWidth = int((H.shape[0] - Nw)/2)
    if Nw % 2 == 0:
        arrayPadder = ([0, 0], [0, 0], [padWidth, padWidth])
    else:
        arrayPadder = ([0, 0], [0, 0], [padWidth, padWidth + 1])

    # Process in chunks so that memory doesn't get overfilled
    for i in range(Niter):
        idx = np.arange(n*i, n*(i+1), dtype=int)
        idx = idx[idx < Nangles]

        t0 = lam.utils.timerStart()
        # convert sinogram to cupy array if it isn't already one
        if type(sinogram) != type(cp.array(0)):
            tmpSinogram = cp.array(sinogram[idx])
        else:
            tmpSinogram = sinogram[idx]
        lam.utils.timerEnd(t0, "applyFilter- move to GPU", timerOn)

        t0 = lam.utils.timerStart()
        # Zero pad projections, important to avoid negative values in 
        # air
        tmpSinogram = cp.pad(tmpSinogram, arrayPadder, 'symmetric')
        lam.utils.timerEnd(t0, "applyFilter- pad", timerOn)

        t0 = lam.utils.timerStart()
        # sinogram holds fft of projections
        with scipy.fft.set_backend(cufft):
            tmpSinogram = scipy.fft.fft(tmpSinogram, axis=2)
        lam.utils.timerEnd(t0, "applyFilter- fft", timerOn)

        # frequency domain filtering
        t0 = lam.utils.timerStart()
        tmpSinogram = tmpSinogram*cp.array(H[:, idx].transpose()[:, np.newaxis])
        lam.utils.timerEnd(t0, "applyFilter- multiply by filter", timerOn)

        t0 = lam.utils.timerStart()
        with scipy.fft.set_backend(cufft):
            tmpSinogram = scipy.fft.ifft(tmpSinogram, axis=2)
        tmpSinogram = cp.real(tmpSinogram)
        lam.utils.timerEnd(t0, "applyFilter- ifft", timerOn)

        t0 = lam.utils.timerStart()
        # Truncate the filtered projections
        truncIdx = np.arange(m/2 - Nw/2, m/2 + Nw/2, dtype=int)
        tmpSinogram = tmpSinogram[:, :, truncIdx]
        filteredSinogram[idx] = tmpSinogram.get()
        lam.utils.timerEnd(t0, "applyFilter- final", timerOn)

    return filteredSinogram
