import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
import cupyx.scipy.interpolate
import cupyx as cpx
import matplotlib.pyplot as plt
import time as time
import copy
import psutil
from pympler import asizeof
import os
import h5py as h5
import pickle
import traceback

timerOn = False


def fixStackDims(stackObject, dims=None, padType='constant'):

    if dims is None:
        [Ny, Nx] = stackObject[0].shape
    else:
        [Ny, Nx] = dims

    print("Fixing stack dimensions to:", "(" + str(Ny) + "," + str(Nx) + ")")
    
    for i in range(len(stackObject)):
        [ny, nx] = np.shape(stackObject[i])
        # Crop or pad rows and columns
        if ny > Ny:
            # Crop rows
            w = ny - Ny
            idx = np.arange(w/2, ny - w/2, dtype=int)
            stackObject[i] = stackObject[i][idx, :]
        elif ny < Ny:
            # Pad rows
            w = Ny - ny
            padWidth = ((int(np.ceil(w/2)), int(np.floor(w/2))), (0, 0))
            stackObject[i] = np.pad(stackObject[i], padWidth, padType)
        if nx > Nx:
            # Crop columns
            w = nx - Nx
            idx = np.arange(w/2, nx - w/2, dtype=int)
            stackObject[i] = stackObject[i][:, idx]
        elif nx < Nx:
            # Pad columns
            w = Nx - nx
            padWidth = ((0, 0), (int(np.ceil(w/2)), int(np.floor(w/2))))
            stackObject[i] = np.pad(stackObject[i], padWidth, padType)

    return stackObject


def cropPad(img, Ny, Nx):

    [ny, nx] = img.shape[1:]

    # Crop as necessary
    if ny > Ny:
        # Crop rows
        w = ny - Ny
        a, b = int(w/2), int(ny - w/2)
        img = img[:, a:b]
    if nx > Nx:
        # Crop columns
        w = nx - Nx
        a, b = int(w/2), int(nx - w/2)
        img = img[:, :, a:b]

    # Pad as necessary
    if ny < Ny:
        # Pad rows
        w = Ny - ny
        padWidth = ((0, 0), (int(np.ceil(w/2)), int(np.floor(w/2))), (0, 0))
        img = np.pad(img, padWidth, mode='constant')
    if nx < Nx:
        # Pad columns
        w = Nx - nx
        padWidth = ((0, 0), (0, 0), (int(np.ceil(w/2)), int(np.floor(w/2))))
        img = np.pad(img, padWidth, mode='constant')

    return img


def rotateStackMod90(img, theta):

    numRotations = round(theta/90)
    if np.ndim(img) == 3:
        img = np.rot90(img, numRotations, axes=(1, 2))
    elif np.ndim(img) == 2:
        img = np.rot90(img)
    else:
        raise Exception("Incompatible dimensions")
    theta = theta - 90*numRotations

    return img


def fftRotate(img, theta):
    """FFT based image rotation for a stack of images"""

    xp = cp.get_array_module(img)

    # For compatibility with both 2 and 3 dimensional arrays
    nDims = xp.ndim(img)
    if nDims == 3:
        c = 1
        [M, N] = np.shape(img)[1:3]
    elif nDims == 2:
        c = 0
        [M, N] = np.shape(img)
    else:
        raise Exception("Incompatible image size")

    isReal = not (img.dtype == np.complex64 or img.dtype == np.complex128)

    numRotations = round(theta/90)
    theta = theta - 90*numRotations

    xgrid = xp.array(
        np.fft.ifftshift(
            np.matrix(
                np.arange(-np.fix(M / 2),
                          np.ceil(M / 2),
                          dtype=np.float32)).transpose()) / M)
    ygrid = xp.array(
        np.fft.ifftshift(
            np.matrix(
                np.arange(-np.fix(N/2),
                          np.ceil(N/2),
                          dtype=np.float32))) / N)

    Mgrid = xp.array((np.matrix(
        np.arange(1, M+1, dtype=np.float32)).transpose() - np.floor(M/2) - 0.5))
    Ngrid = xp.array((np.matrix(
        np.arange(1, N+1, dtype=np.float32)) - np.floor(N/2) - 0.5))

    Nx = xp.array(-np.sin(theta * np.pi/180) * xgrid, dtype=np.float32)
    Ny = xp.array(np.tan(theta/2 * np.pi/180) * ygrid, dtype=np.float32)

    M1 = xp.array(np.exp(-2j*np.pi * Mgrid * Ny), dtype=np.complex64)
    M2 = xp.array(np.exp(-2j*np.pi * xp.multiply(Ngrid, Nx)), dtype=np.complex64)

    if xp.__name__ == 'numpy':
        backend = 'scipy'
    else:
        backend = cufft

    # rotate images by a combination of shears
    with scipy.fft.set_backend(backend):
        img = scipy.fft.ifft(xp.multiply(scipy.fft.fft(img, axis=c+1), M1), axis=c+1)
        img = scipy.fft.ifft(xp.multiply(scipy.fft.fft(img, axis=c), M2), axis=c)
        img = scipy.fft.ifft(xp.multiply(scipy.fft.fft(img, axis=c+1), M1), axis=c+1)

    if isReal:
        img = np.real(img)

    return img


def fftImageShear(img, theta):
    """FFT based image shear for a stack of images"""

    xp = cp.get_array_module(img)

    # Included so this works with 2 and 3 dimensional arrays
    nDims = xp.ndim(img)
    if nDims == 3:
        c = 1
        [M, N] = np.shape(img)[1:3]
    elif nDims == 2:
        c = 0
        [M, N] = np.shape(img)
    else:
        raise Exception("Incompatible image size")

    Nx = xp.array(
        scipy.fft.ifftshift(np.arange(-np.fix(M/2), np.ceil(M/2)))
        * -np.sin(theta * np.pi/180)/M)
    Ny = xp.array(
        scipy.fft.ifftshift(np.arange(-np.fix(N/2), np.ceil(N/2)))
        * np.tan(theta * np.pi/180/2)/N)
    Mgrid = xp.array(
        (np.matrix(
            np.arange(1, M+1) - np.floor(M/2))
            .transpose()*2j*np.pi))
    Ngrid = xp.array(
        (np.matrix(
            np.arange(1, N+1) - np.floor(N/2))
            .transpose()*2j*np.pi))

    if xp.__name__ == 'numpy':
        backend = 'scipy'
    else:
        backend = cufft

    # Rotate images by a combination of shears
    with scipy.fft.set_backend(backend):
        img = scipy.fft.ifft(
            np.multiply(scipy.fft.fft(img, axis=c+1), np.exp(-np.multiply(Mgrid, Ny))),
            axis=c+1)

    return img


def initialImageProcess(img, thetaRotate, thetaShear, rotationType=1):
    if rotationType == 1:
        img = fftRotate(img, theta=thetaRotate)
    else:
        numRotations = round(thetaRotate/90)
        thetaRotate = thetaRotate - 90*numRotations
        img = cpx.scipy.ndimage.rotate(img, thetaRotate, reshape=False, axes=(1, 2))
    img = fftImageShear(img, theta=thetaShear)
    return img


def fftImShift(img, shift, applyFFT=True):

    t00 = timerStart()

    x = shift[:, 0][:, np.newaxis]
    y = shift[:, 1][:, np.newaxis]

    t0 = timerStart()
    isReal = not (img.dtype == np.complex64 or img.dtype == np.complex128)
    timerEnd(t0, "fftImShift- check if real", timerOn)

    Np = np.shape(img)

    if applyFFT:
        t0 = timerStart()
        with scipy.fft.set_backend(cufft):
            img = scipy.fft.fft2(img)
        timerEnd(t0, "fftImShift- apply FFT", timerOn)

    t0 = timerStart()
    xGrid = scipy.fft.ifftshift(
        np.arange(-np.fix(Np[2]/2), np.ceil(Np[2]/2), dtype=np.float32)
    )/Np[2]
    X = (x*xGrid)[:, np.newaxis, :]
    X = np.exp(-2j*np.pi*X)

    yGrid = scipy.fft.ifftshift(
        np.arange(-np.fix(Np[1]/2), np.ceil(Np[1]/2), dtype=np.float32)
    )/Np[1]
    Y = (y*yGrid)[:, :, np.newaxis]
    Y = np.exp(-2j*np.pi*Y)
    timerEnd(t0, "fftImShift- get X and Y", timerOn)

    img = img*cp.array(X)
    img = img*cp.array(Y)
    timerEnd(t0, "fftImShift- multiply img by X and Y", timerOn)

    t0 = timerStart()
    if applyFFT:
        with scipy.fft.set_backend(cufft):
            img = scipy.fft.ifft2(img)
        timerEnd(t0, "fftImShift- apply inverse FFT", timerOn)

    t0 = timerStart()
    if isReal:
        img = np.real(img)
    timerEnd(t0, "fftImShift- convert to real", timerOn)

    timerEnd(t00, "fftImShift- total", timerOn)
    return img


def plotScaledImage(img, scale):

    ax = plt.imshow(np.real(img), cmap='viridis')
    plt.colorbar()
    ax.set_clim(scale[0], scale[1])


def stabilizePhase(img, weights, imgRef=1, normalizeAmplitude=True, removeRamp=True,
                   fourierGuess=True):

    try:
        # Change this if you end up adding binning as a feature
        binning = 1
        phaseDiff = imgRef*cp.conj(img)
        [M, N] = cp.shape(phaseDiff)[1:3]
        xRamp = cp.pi*cp.linspace(-1, 1, M).reshape(1, M, 1)
        yRamp = cp.pi*cp.linspace(-1, 1, N).reshape(1, 1, N)

        if removeRamp and fourierGuess:
            xCorrMatrix = scipy.fft.ifftshift(cp.abs(scipy.fft.ifft2(phaseDiff)))**2
            # cp.unravel_index takes the longest
            [x, y] = cp.unravel_index(cp.argmax(xCorrMatrix, axis=(1, 2)), xCorrMatrix.shape[1:3])
            x = (x - np.ceil(M/2)).reshape(len(x), 1, 1)
            y = (y - np.ceil(N/2)).reshape(len(y), 1, 1)
            cOffset = xRamp*x + yRamp*y
            phaseDiff = phaseDiff*cp.exp(1j*cOffset)

        # Calculate optimal phase shift
        gamma = np.mean(phaseDiff*weights, axis=(1, 2))/np.mean(weights)
        gamma = (gamma/cp.abs(gamma)).reshape(len(gamma), 1, 1)

        if removeRamp:
            phaseDiff = phaseDiff*gamma.conj()
            # Linear refinement
            phaseDiff = cp.angle(phaseDiff)*weights
            xGamma = (cp.mean(phaseDiff*xRamp, axis=(1, 2))
                      / cp.mean(weights*cp.abs(xRamp)**2)).reshape(len(x), 1, 1)
            yGamma = (cp.mean(phaseDiff*yRamp, axis=(1, 2))
                      / cp.mean(weights*cp.abs(yRamp)**2)).reshape(len(y), 1, 1)

            if fourierGuess:
                # Get total correction
                xGamma = xGamma - x
                yGamma = yGamma - y

        # Export dimensionless
        xGamma = xGamma/M
        yGamma = yGamma/N

        # Correct the output image
        def removePhase(imgOrig, xRamp, yRamp, gamma, xGamma, yGamma):
            cOffset = cp.angle(gamma) + xRamp*xGamma + yRamp*yGamma
            imgOut = imgOrig*cp.exp(1j*cOffset)
            return imgOut
        imgOut = removePhase(img, xRamp, yRamp, gamma, xGamma*M/binning, yGamma*N/binning)

        # Normalize amplitude
        if normalizeAmplitude:
            meanAmpl = (weights*imgOut).mean()/weights.mean()
            imgOut = imgOut/meanAmpl

        return imgOut, gamma, xGamma, yGamma

    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        traceback.print_exc()


def addToProjection(smallArray, fullArray, positionOffset):

    xp = cp.get_array_module(fullArray)

    Nf = np.shape(fullArray)
    Ns = np.shape(smallArray)
    idx_f = {}
    idx_s = {}
    for ii in range(len(positionOffset)):
        # Only set up for 2D arrays right now
        for i in range(2):
            idx_f[i] = np.arange(np.fmax(0, positionOffset[ii, i]).max(),
                                 np.fmin(Nf[i], positionOffset[ii, i] + Ns[i]).min(),
                                 dtype='int')
            idx_s[i] = np.arange(idx_f[i][0] - positionOffset[ii, i],
                                 idx_f[i][-1] - positionOffset[ii, i] + 1,
                                 dtype='int')
        fullArray[np.ix_(idx_f[0], idx_f[1])] = (fullArray[np.ix_(idx_f[0], idx_f[1])]
                                                 + smallArray[np.ix_(idx_s[0], idx_s[1])])
    return fullArray


def getROI(mask, extent=0, multiplyOf=1):
    # get_ROI

    x = np.any(mask, 1)
    y = np.any(mask, 0)
    coord = np.array([np.argwhere(x)[0][0], np.argwhere(x)[-1][0],
                      np.argwhere(y)[0][0], np.argwhere(y)[-1][0]])

    w = coord[1] - coord[0]
    h = coord[3] - coord[2]
    Cx = (coord[1] + coord[0])/2
    Cy = (coord[3] + coord[2])/2

    coord[0] = np.floor(Cx - np.ceil((0.5 + extent)*w))
    coord[1] = np.ceil(Cx + np.ceil((0.5 + extent)*w))

    coord[2] = np.floor(Cy - np.ceil((0.5+extent)*h))
    coord[3] = np.ceil(Cy + np.ceil((0.5+extent)*h))

    w = np.floor((coord[1] - coord[0])/multiplyOf)*multiplyOf
    h = np.floor((coord[3] - coord[2])/multiplyOf)*multiplyOf

    coord[1] = coord[0] + w - 1
    coord[3] = coord[2] + h - 1
    coord[[0, 2]] = np.max([np.zeros((2)), coord[[0, 2]]], axis=0)
    coord[[1, 3]] = np.min([mask.shape - np.array([1, 1]), coord[[1, 3]]], axis=0)

    ROI = {}
    ROI[0] = np.arange(coord[0], coord[1] + 1, dtype=int)
    ROI[1] = np.arange(coord[2], coord[3] + 1, dtype=int)

    return ROI


def findEdge(arr, axis=0, direction=1):
    """Find the edges of mask in weightSino"""
    # Perhaps move to the 'tomo' file

    xp = cp.get_array_module(arr)

    # Step sizes for searching for first non-zero element
    sizes = [250, 125, 50]
    startIdx = 0
    for size in sizes:
        spc = size*direction
        rng = np.arange(startIdx, arr.shape[axis], dtype=int)[::spc]
        for i in rng:
            if xp.any(arr.take(indices=i, axis=axis)):
                startIdx = i - size
                break
    # Estimated edge location
    idx = startIdx + size

    # Make sure edges are fully enclosed (you will cut some non-zero 
    # area off, but thats ok)
    idx = idx - size*direction
    if idx < 0:
        idx = 0
    elif idx >= arr.shape[axis]:
        idx = arr.shape[axis] - 1

    return idx


def imShiftLinear(img, shift, method='circ', n=0, downsample=[1, 1], cutEdge=False):

    if not hasattr(downsample, '__len__'):
        downsample = [downsample, downsample]
    elif len(downsample) == 1:
        downsample = [downsample[0], downsample[0]]
    elif type(downsample) is np.ndarray:
        downsample = list(downsample)

    xp = cp.get_array_module(img)
    if xp is cp:
        interpolator = cpx.scipy.interpolate.RegularGridInterpolator
    else:
        interpolator = scipy.interpolate.RegularGridInterpolator

    x = shift[:, 0]
    y = shift[:, 1]

    Nz, Nx, Ny = img.shape
    X = xp.arange(0, Nx, dtype=int)
    Y = xp.arange(0, Ny, dtype=int)

    if method == 'circ':
        if cutEdge:
            # find edges so that you can cut out unwanted wrap-around
            bounds = np.zeros((len(img), 2, 2), dtype=int)
            for i in range(len(img)):
                bounds[i] = [[findEdge(img[i], 1, 1), findEdge(img[i], 1, -1)],
                             [findEdge(img[i], 0, 1), findEdge(img[i], 0, -1)]]
        for ii in range(Nz):
            xIdx = xp.roll(X, int(xp.round(y[ii])))
            yIdx = xp.roll(Y, int(xp.round(x[ii])))
            img[ii] = img[ii, xIdx[:, np.newaxis], yIdx[np.newaxis]]

        if cutEdge:
            # cut off wrap-around
            L = np.array(img.shape[1:][::-1])

            def getRemovalIdx(bounds, L):
                if bounds[0] < bounds[1]:
                    removalIdx = []
                elif bounds[0] < (L - bounds[1]):
                    removalIdx = np.arange(0, bounds[0], dtype=int)
                else:
                    removalIdx = np.arange(bounds[0], L, dtype=int)
                return removalIdx

            for i in range(len(img)):
                hBounds = (bounds[i][0] + shift[i][0]) % L[0]
                vBounds = (bounds[i][1] + shift[i][1]) % L[1]
                hIdx = getRemovalIdx(hBounds, L[0])
                vIdx = getRemovalIdx(vBounds, L[1])
                img[i][vIdx] = 0
                img[i][:, hIdx] = 0

    elif method == 'linear':
        if n == 0:
            n = Nz
        Niter = int(np.ceil(Nz/n))
        if downsample != [1, 1]:
            newNx = int(round(Nx/downsample[0]))
            newNy = int(round(Ny/downsample[1]))
            imgNew = xp.zeros((Nz, newNx, newNy))
        else:
            newNx = Nx
            newNy = Ny
            imgNew = img
        for i in range(Niter):
            idx = np.arange(n*i, n*(i+1), dtype=int)
            idx = idx[idx < Nz]
            # Create the interpolation function
            x0 = xp.arange(0, Nx, dtype=int)
            y0 = xp.arange(0, Ny, dtype=int)
            z0 = xp.arange(0, len(idx), dtype=int)
            interpFunc = interpolator(
                (z0, x0, y0), img[idx], bounds_error=False, fill_value=0
            )
            # Define the new coordinates
            x0 = np.linspace(x0[0], x0[-1], newNx, dtype=int)
            y0 = np.linspace(y0[0], y0[-1], newNy, dtype=int)

            Z, X, Y = xp.meshgrid(z0, x0, y0, indexing='ij')
            X = X + xp.array(-shift[idx, 1])[:, xp.newaxis][:, :, xp.newaxis]
            Y = Y + xp.array(-shift[idx, 0])[:, xp.newaxis][:, :, xp.newaxis]

            # Get the interpolated function at the new coordinates
            imgNew[idx] = interpFunc((Z, X, Y))
        img = imgNew

    return img


def interpolateFT(img, outsize):

    Nout = outsize
    Nin = img.shape
    with scipy.fft.set_backend(cufft):
        imFT = scipy.fft.fftshift(scipy.fft.fft2(img), (1, 2))
        imOut = cropPad(imFT, Nout[0], Nout[1])
        imOut = scipy.fft.ifftshift(imOut, (1, 2))
        imOut = scipy.fft.ifft2(imOut)*((Nout[0]*Nout[1])/(Nin[1]*Nin[2]))

    return imOut


def interpLinear(img, sizeOut):

    Nlayers, Nx, Ny = np.shape(img)
    imgOut = np.zeros((Nlayers, sizeOut[0], sizeOut[1]), like=img)

    for ii in range(Nlayers):
        interpSpline = scipy.interpolate.RectBivariateSpline(
            x=np.arange(0, Nx),
            y=np.arange(0, Ny),
            z=img[ii].get())
        imgOut[ii] = cp.array(interpSpline(
            x=np.linspace(0, Nx, sizeOut[0]),
            y=np.linspace(0, Ny, sizeOut[1]).transpose()))

    return imgOut


def removeSinogramRamp(sinogram, airGap, polyfitOrder):

    airGap = np.ceil(airGap)
    Nlayers = sinogram.shape[1]
    widthSinogram = sinogram.shape[2]
    ax = np.arange(0, widthSinogram)
    mask = {}
    mask[0] = cp.array(ax <= airGap[0])
    mask[1] = cp.array(ax >= widthSinogram - airGap[0])

    airValues = {}
    for ii in range(2):
        # get average values in the air_gap
        airValues[ii] = cp.sum(sinogram*mask[ii], axis=2)/cp.sum(mask[ii])
        # find 2D plane that fits the air region
        ramp = cp.linspace(-1, 1, Nlayers)[np.newaxis, :]
        # Iteratively refine the ideal plane estimate to ignore 
        # imperfect estimations of the air_gap parameter
        weight = 1
        for jj in range(10):
            # Fit it with a linear plane, use for 2D unwrapping
            planeFit = (
                cp.mean(weight*airValues[ii], axis=1)[:, np.newaxis]
                / np.mean(weight)
                + np.mean(weight*airValues[ii]*ramp, axis=1)[:, np.newaxis]
                / np.mean(weight*ramp**2)*ramp
            )
            deviation = 5*cp.mean(cp.abs(
                (airValues[ii] - planeFit) - cp.mean(airValues[ii] - planeFit))
            )
            # avoid outliers by adaptive weighting
            weight = 1/(1 + (cp.abs(airValues[ii] - planeFit)/deviation)**2)
        airValues[ii] = planeFit
    # get a plane in between left and right side passing through
    interpFunction = cupyx.scipy.interpolate.BarycentricInterpolator(
        cp.array([0, widthSinogram]),
        cp.append(airValues[0][:, :, np.newaxis].get(),
                  airValues[1][:, :, np.newaxis].get(), axis=2).transpose([2, 0, 1])
    )
    ramp = interpFunction(cp.arange(0, widthSinogram, dtype=int)).transpose([1, 2, 0])
    sinogram = sinogram - ramp

    return sinogram


def interpFT_centered(img, NpNew, interpSign=-1):
    # interpolateFT_centered
    # Beware: Often gives errors when NpNew are not even numbers

    padBy = 2  # Must be even

    t0 = timerStart()
    Np = np.array(img.shape)[1:]
    NpNew = NpNew + padBy
    isReal = not (img.dtype == np.complex64 or img.dtype == np.complex128)

    scale = np.prod(NpNew - padBy)/np.prod(Np)
    downsample = int(np.ceil(np.sqrt(1/scale)))

    # apply the padding to account for boundary issues
    padWidth = int(downsample*padBy/2)
    padShape = cp.pad(img[0], padWidth, 'symmetric').shape

    imgPad = cp.zeros((len(img), padShape[0], padShape[1]), dtype=img.dtype)
    for i in range(len(img)):
        imgPad[i] = np.pad(img[i], padWidth, 'symmetric')
    img = imgPad
    del imgPad

    # go to the fourier space
    with scipy.fft.set_backend(cufft):
        img = scipy.fft.fft2(img)
    timerEnd(t0, "interpFT_centered- Apply FFT", timerOn)

    # apply +/-0.5 px shift # this is the slowest!
    t0 = timerStart()
    img = fftImShift(img, np.array([[interpSign*-0.5, interpSign*-0.5]]), applyFFT=False)
    timerEnd(t0, "interpFT_centered- Uncropped FFT shift", timerOn)

    # crop in the Fourier space
    t0 = timerStart()
    with scipy.fft.set_backend(cufft):
        img = scipy.fft.ifftshift(
            cropPad(scipy.fft.fftshift(img, axes=(1, 2)), NpNew[0], NpNew[1]), axes=(1, 2))
    timerEnd(t0, "interpFT_centered- Crop in Fourier Space", timerOn)

    t0 = timerStart()
    # apply -/+0.5 px shift in the cropped space
    img = fftImShift(img, np.array([[interpSign*0.5, interpSign*0.5]]), applyFFT=False)
    timerEnd(t0, "interpFT_centered- Shift in cropped space", timerOn)

    # return to the real space
    t0 = timerStart()
    with scipy.fft.set_backend(cufft):
        img = scipy.fft.ifft2(img)
    timerEnd(t0, "interpFT_centered- Apply iFFT", timerOn)

    t0 = timerStart()
    # scale to keep the average constant
    img = img*scale
    timerEnd(t0, "interpFT_centered- Scale image", timerOn)

    t0 = timerStart()
    # remove the padding
    a = int(padBy/2)
    img = img[:, a:(NpNew[0] - a), a:(NpNew[1] - a)]
    timerEnd(t0, "interpFT_centered- Remove padding", timerOn)

    t0 = timerStart()
    if isReal:
        img = cp.real(img)
    timerEnd(t0, "interpFT_centered- Convert to real", timerOn)

    return img


def imShiftGeneric(img, shift=0, ROI=[], downsample=1, interpMethod='fft', interpSign=-1,
                   downsampleMethod=1):
    """Downsample and shift the input array stack"""

    t00 = timerStart()
    # Shift the sinogram
    if np.any(shift != 0):
        if interpMethod == 'fft':
            img = fftImShift(img, shift)
        elif interpMethod == 'linear':
            img = imShiftLinear(img, shift, 'linear')

    isReal = not (img.dtype == np.complex64 or img.dtype == np.complex128)
    Np = np.array(img.shape)[1:]

    # Downsample the sinogram using interpolation
    if downsample > 1:
        for i in range(len(img)):
            img[i, :, :] = cpx.scipy.ndimage.gaussian_filter(img[i, :, :], downsample)
        img = img/cpx.scipy.ndimage.gaussian_filter(cp.ones(Np, np.float32), downsample)

        if interpMethod == 'fft':
            t0 = timerStart()
            # NpNew must be even to avoid errors in interpFT_centered
            NpNew = np.round(np.ceil(Np/downsample/2)*2)
            img = interpFT_centered(img, NpNew, interpSign)
            timerEnd(t0, "imshift_generic- interpFT_centered")
        elif interpMethod == 'linear':
            # img = interpFT_centered(img, np.round(Np/downsample), interpSign)
            # img = imShiftLinear(img, shift, "linear")
            pass

    if isReal:
        img = cp.real(img)
    timerEnd(t00, "imShiftGeneric- Total", timerOn)

    return img


def radtap(X, Y, tappix, zerorad):
    tau = 2*tappix
    R = np.sqrt(X**2 + Y**2)
    taperfunc = 0.5*(1 + np.cos(2*np.pi*(R - zerorad - tau/2)/tau))
    taperfunc = (R > zerorad + tau/2)*1.0 + taperfunc*(R <= zerorad + tau/2)
    taperfunc = taperfunc*(R >= zerorad)

    return taperfunc


def apply3DApodization(tomogram, radApod, axialApod, radialSmooth):
    Npix = tomogram.shape[0]
    xt = np.arange(-Npix/2, Npix/2)
    X, Y = np.meshgrid(xt, xt)
    if len(np.shape(radialSmooth)) > 0:
        radialSmooth[radialSmooth < 1] = 1  # prevent division by zero
    circulo = 1 - radtap(X, Y, radialSmooth, np.round(Npix/2 - radApod - radialSmooth))
    tomogram = tomogram*circulo[:, :, np.newaxis]

    return circulo


def imFilterHighPass1D(img, ax, sigma, applyFFT=True):

    xp = cp.get_array_module(img)
    if xp.__name__ == 'numpy':
        backend = 'scipy'
    else:
        backend = cufft

    Npix = img.shape
    isReal = not (img.dtype == np.complex64 or img.dtype == np.complex128)

    if applyFFT:
        with scipy.fft.set_backend(backend):
            img = scipy.fft.fft(img, axis=ax)

    x = xp.arange(-Npix[ax]/2, Npix[ax]/2, dtype=np.float32)/Npix[ax]

    # Solution to make the filter resolution independent
    # For different levels of scaling, the filtered field will look the 
    # same
    sigma = 256/Npix[ax]*sigma

    if sigma == 0:
        # Not utilized
        # spectral_filter = 2i*pi*(fftshift((0:Npix(ax)-1)/Npix(ax))-0.5)
        pass
    else:
        spectralFilter = scipy.fft.fftshift(xp.exp(1/(-x**2/sigma**2)))

    shape = [1, 1, 1]
    shape[ax] = Npix[ax]
    img = img*spectralFilter.reshape(shape)

    if applyFFT:
        with scipy.fft.set_backend(backend):
            img = scipy.fft.ifft(img, axis=ax)
    if isReal:
        img = xp.real(img)

    return img


def printUse(mempool=cp.get_default_memory_pool()):
    bytes_to_MiB = 1/1048576
    print(cp.cuda.Device())
    print("   Used:", round(mempool.used_bytes()*bytes_to_MiB), 'MiB')
    print("  Total:", round(mempool.total_bytes()*bytes_to_MiB), 'MiB')


def freeAllBlocks(devices=[cp.cuda.runtime.getDevice()]):
    for i in devices:
        with cp.cuda.Device(i):
            cp.get_default_memory_pool().free_all_blocks()


def timerStart():
    t0 = time.time()
    return t0


def timerEnd(t0, title="", timerOn=timerOn, subtract=True):
    if timerOn:
        if subtract:
            t = time.time()-t0
        else:
            t = t0
        print("{:.4f}".format(t), "s ", title)
        return t


def getTotalMemUsageCPU():
    # Get available memory
    virtualMemory = psutil.virtual_memory()
    availableMemoryGB = virtualMemory.available / (1024 ** 3)  # Convert to gigabytes
    print(f"Available Memory: {availableMemoryGB:.2f} GB")
    # Get used memory
    virtualMemory = psutil.virtual_memory()
    usedMemoryGB = virtualMemory.used / (1024 ** 3)  # Convert to gigabytes
    print(f"Used Memory: {usedMemoryGB:.2f} GB")


def getVariableMemUsageCPU(variable):
    sizeInBytes = asizeof.asizeof(variable)
    sizeInGB = sizeInBytes / (1024 ** 3)  # Convert to gigabytes
    print(f"Total memory size of variable: {sizeInGB:.2f} GB")


def saveDictAsH5(d, filePath):
    """Save a dict as an h5 file"""

    with h5.File(filePath, 'w') as hf:
        for k in d.keys():
            print('saving ' + str(k))
            hf.create_dataset(str(k), data=d[k])
    print(f"Data saved to {filePath}")


def saveArrays(obj, saveFolder, fileName, exclude=[]):
    """Save any array that has 3 or more dimensions"""

    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    filePath = os.path.join(saveFolder, fileName)

    saveAttrs = [attr for attr in obj.__dict__.keys()
                 if type(getattr(obj, attr)) is np.ndarray
                 and len(np.shape(getattr(obj, attr))) > 2
                 and attr not in exclude]

    # Create an HDF5 file
    with h5.File(filePath, 'w') as hf:
        for name in saveAttrs:
            print('saving ' + name + ' to ' + fileName)
            if getattr(obj, name).dtype != np.complexfloating:
                hf.create_dataset(name, data=getattr(obj, name))
            else:
                hf.create_dataset(name + '_real', data=np.real(getattr(obj, name)))
                hf.create_dataset(name + '_imag', data=np.imag(getattr(obj, name)))
    print(f"Data saved to {filePath}")


def partialPickle(obj, saveFolder, fileName, exclude=[]):
    """Pickles a data object, but excludes any array or list with 3 or 
    more dimensions"""

    if type(exclude) != list:
        exclude = [exclude]

    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    filePath = os.path.join(saveFolder, fileName)

    attrs = set(obj.__dict__.keys())

    # Exclude specified attributes and any attributes that are arrays 
    # with 3 or more dims
    exclude = exclude + [attr for attr in obj.__dict__.keys()
                         if type(getattr(obj, attr)) is np.ndarray
                         and len(np.shape(getattr(obj, attr))) > 2]

    fileObj = open(filePath, 'wb')
    pickle.dump({k: getattr(obj, k) for k in attrs if k not in exclude}, fileObj)
    fileObj.close()


def getRotationCenter(sinogram, CoR):

    rotationCenter = np.array(sinogram.shape[1:])/2 + CoR[[1, 0]]
    return rotationCenter
