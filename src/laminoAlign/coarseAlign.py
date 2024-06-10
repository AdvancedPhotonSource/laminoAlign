import laminoAlign as lam
import numpy as np
import cupy as cp
import scipy
import cupyx.scipy.fft as cufft
from cupyx.scipy.signal import fftconvolve
import cupyx as cpx
import statsmodels.robust
from IPython.display import clear_output
from scipy.signal import savgol_filter
import pandas as pd

timerOn = True


def alignTomoXCorr(stackObject, illumSum, angles, xCorrROI, blockLength, parameters):

    # Get variation field
    weights = cp.array(illumSum / (illumSum + 0.1*np.max(illumSum)))
    weights = weights[xCorrROI[0]][:, xCorrROI[1]]
    stackObject = stackObject[:, xCorrROI[0]][:, :, xCorrROI[1]]
    variation = cp.zeros((len(stackObject),
                          int(np.ceil(xCorrROI[0].shape[0]/parameters['binning'])),
                          int(np.ceil(xCorrROI[1].shape[0]/parameters['binning']))),
                         dtype=np.float32)

    numBlocks = np.arange(0, np.ceil(len(stackObject)/blockLength), dtype='int')
    n = blockLength
    Nangles = len(stackObject)

    for i in numBlocks:
        t0 = lam.utils.timerStart()
        idx = np.arange(n*i, n*(i+1), dtype=int)
        idx = idx[idx < Nangles]

        variation[idx, :, :] = getVariationField(
            cp.array(stackObject[idx]),
            weights,
            parameters['binning'])
        lam.utils.timerEnd(t0, "getVariationField loop", timerOn=timerOn)
        clear_output(wait=True)
    lam.utils.freeAllBlocks()

    sortIdx = np.argsort(angles)
    sortInvIdx = np.argsort(sortIdx)

    # Get initial guess of the relative shifts
    Nangles = len(variation)
    totalShift = np.zeros((Nangles, 2))
    savedShifts = np.zeros((parameters['maxIter'], Nangles, 2))

    for iter in range(parameters['maxIter']):
        t0 = lam.utils.timerStart()

        fvar = filteredFFT(variation, parameters, totalShift.astype(np.float32))
        lam.utils.freeAllBlocks()

        relativeShifts = findShiftFast(fvar[sortIdx], fvar[np.roll(sortIdx, -1)])
        relativeShifts = np.roll(relativeShifts, 1, 0)

        # Avoid too fast jumps by limiting the maximal step size per 
        # iteration
        maxShift = 3*statsmodels.robust.mad(relativeShifts, axis=0)
        maxShift[maxShift < 10] = 10
        sign = np.sign(relativeShifts)
        for i in range(2):
            sign = np.sign(relativeShifts[:, i])
            idx = maxShift[i] < np.abs(relativeShifts[:, i])
            relativeShifts[idx, i] = maxShift[i]*sign[idx]

        # Long term drifts cannot be trusted
        cumShift = np.cumsum(relativeShifts, axis=0)
        cumShift = cumShift - np.mean(cumShift, axis=0)

        # Minimize the shift amplitude that is needed
        for i in range(2):
            # Get properly smoothed cumulative shift
            smooth = int(np.ceil(parameters['filterPos']/2)*2 + 1)

            df = pd.DataFrame(dict(x=cumShift[:, i]))
            smoothedShift = df[["x"]].apply(savgol_filter,
                                            window_length=parameters['filterPos'],
                                            polyorder=2).to_numpy()[:, 0]

            # Subtract from cumShift
            cumShift[:, i] = cumShift[:, i] - smoothedShift

        totalShift = totalShift + cumShift[sortInvIdx, :]

        # Limit the maximal shift to 3*(mean absolute deviation of all 
        # the positions). This prevents outliers.
        for i in range(2):
            idx = 6*statsmodels.robust.mad(totalShift, axis=0)[i] < np.abs(totalShift[:, i])
            sign = np.sign(totalShift[:, i])
            totalShift[idx, i] = 6*statsmodels.robust.mad(totalShift, axis=0)[i]*sign[idx]
            # Save total shifts for plotting
        savedShifts[iter, :, :] = totalShift

        clear_output(wait=True)
        print("iteration " + str(iter + 1) + "/" + str(parameters['maxIter']))
        lam.utils.timerEnd(t0, "Loop time", timerOn=True)

    # Return shifted projections for user to judge the alignment
    # quality
    variationAligned = lam.utils.fftImShift(variation, totalShift)
    totalShift = np.round(totalShift*parameters['binning'])

    return totalShift, variation, variationAligned, savedShifts


def filteredFFT(img, parameters, shift):
    [nx, ny] = img.shape[1:3]

    img = lam.utils.fftImShift(img, shift)

    spatialFilter = cp.array(scipy.signal.windows.tukey(nx, 0.3)[:, np.newaxis]
                             * scipy.signal.windows.tukey(ny, 0.3)[np.newaxis, :],
                             dtype=np.float32)
    img = img - cp.mean(img)
    img = img*spatialFilter
    with scipy.fft.set_backend(cufft):
        img = scipy.fft.fft2(img)

    # Remove low frequencies (e.g. phase ramp issues)
    if parameters['filterData'] > 0:
        X, Y = cp.meshgrid(cp.arange(-nx/2, nx/2, dtype=np.float32),
                           cp.arange(-ny/2, ny/2, dtype=np.float32))
        spectralFilter = scipy.fft.fftshift(
            cp.exp(-(0.5 * (nx + ny) * parameters['filterData'])**2 / (X**2 + Y**2))
        ).transpose()
        img = img*spectralFilter

    return img


def getVariationField(obj, weights, binning):
    # get_variation_field in tomo/align_tomo_xcorr

    t0 = lam.utils.timerStart()
    dX = fftconvolve(obj, cp.array([[[1, -1]]]), 'same')
    dY = fftconvolve(obj, cp.array([[[1, -1]]]).transpose(0, 2, 1), 'same')

    # Get total variation, you will get better results than raw object
    variation = cp.sqrt(cp.abs(dX)**2 + cp.abs(dY)**2)

    # Ignore regions with low amplitude
    variation = variation * cp.abs(obj)

    # *Skipping removal of phase artifacts

    # crop values exceeding limits, important mainly for laminography 
    # where the field of view can contain weakly illuminated (ie very 
    # noisy) regions
    meanVariation = cp.mean(variation*weights, axis=(1, 2))/cp.mean(weights)
    stdVariation = cp.sqrt(cp.mean(
        (variation - meanVariation[:, cp.newaxis, cp.newaxis])**2 * weights,
        axis=(1, 2))/cp.mean(weights))

    for i in range(len(meanVariation)):
        maxValue = meanVariation[i] + stdVariation[i]
        variation[i, (variation[i, :, :] > maxValue)] = maxValue
        variation[i, :, :] = cpx.scipy.ndimage.gaussian_filter(variation[i, :, :], 2*binning)

    variation = cp.real(variation[:, 0::binning, 0::binning])
    lam.utils.timerEnd(t0, "getVariationField", timerOn=timerOn)

    return variation


def findShiftFast(o1, o2):
    "Fast subpixel cross correlation"

    with scipy.fft.set_backend(cufft):
        xCorrMat = scipy.fft.fftshift(cp.abs(scipy.fft.ifft2(o1 * cp.conj(o2))), (1, 2))
    lam.utils.freeAllBlocks()
    # take only small region around maximum
    win = 5
    kernelSize = [5, 5]
    mask = (cpx.scipy.signal.fftconvolve(
        (xCorrMat == np.max(xCorrMat, axis=(1, 2))[:, np.newaxis, np.newaxis]),
        cp.ones(kernelSize, dtype='single')[np.newaxis], 'same') > .1)
    lam.utils.freeAllBlocks()
    xCorrMat[~mask] = np.inf
    xCorrMat = xCorrMat - np.min(xCorrMat, (1, 2))[:, np.newaxis, np.newaxis]
    xCorrMat[xCorrMat < 0] = 0
    xCorrMat[~mask] = 0
    xCorrMat = (xCorrMat/np.max(xCorrMat, axis=(1, 2))[:, np.newaxis, np.newaxis])**2
    # Get CoM of the central peak only
    xCorrMat = xCorrMat - 0.5
    xCorrMat[xCorrMat < 0] = 0
    xCorrMat = xCorrMat**2

    def findCenterFast(xCorrMat):
        mass = np.sum(xCorrMat, (1, 2))
        N, M = xCorrMat.shape[1:3]
        x = cp.sum(cp.sum(xCorrMat, 1)*cp.arange(0, M), 1)/mass - np.floor(M/2)
        y = cp.sum(cp.sum(xCorrMat, 2)*cp.arange(0, N), 1)/mass - np.floor(N/2)

        return x, y, mass
    x, y, mass = findCenterFast(xCorrMat)
    relativeShifts = cp.array([x, y]).transpose().get()

    return relativeShifts
