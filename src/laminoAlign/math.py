import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
import matplotlib.pyplot as plt
import time as time


def getImgGrad(img):

    Np = cp.shape(img)

    X = 2j*cp.pi*scipy.fft.ifftshift(
        cp.arange(-np.fix(Np[2]/2), np.ceil(Np[2]/2))
    )/Np[2]
    with scipy.fft.set_backend(cufft):
        dX = scipy.fft.fft(img, axis=2)*X
        dX = scipy.fft.ifft(dX, axis=2)

    Y = 2j*cp.pi*scipy.fft.ifftshift(
        cp.arange(-np.fix(Np[1]/2), np.ceil(Np[1]/2))
    )/Np[1]
    with scipy.fft.set_backend(cufft):
        dY = scipy.fft.fft(img, axis=1)*Y[:, np.newaxis]
        dY = scipy.fft.ifft(dY, axis=1)

    return dX, dY


def getPhaseGradient(img):
    dX, dY = getImgGrad(img)
    dX = cp.imag(cp.conj(img)*dX)
    dY = cp.imag(cp.conj(img)*dY)

    return dX, dY


def getImgInt2D(dX, dY):

    Np = dX.shape
    with scipy.fft.set_backend(cufft):
        fD = scipy.fft.fft2(dX + 1j*dY, axes=(1, 2))
    xGrid = scipy.fft.ifftshift(
        cp.arange(-np.fix(Np[2]/2), cp.ceil(Np[2]/2))
    )/Np[2]
    yGrid = scipy.fft.ifftshift(
        cp.arange(-np.fix(Np[1]/2), np.ceil(Np[1]/2))
    )/Np[1]

    X = cp.exp((2j*cp.pi) * xGrid + yGrid[:, cp.newaxis])
    # apply integration filter
    X = X/(2j*cp.pi*(xGrid + 1j*yGrid[:, cp.newaxis]))
    X[0, 0] = 0
    integral = fD*X
    with scipy.fft.set_backend(cufft):
        integral = scipy.fft.ifft2(integral, axes=(1, 2))

    return integral


def unwrap2Dfft(img, weights):

    step = 0
    emptyRegion = []

    weights[weights > 1] = 1
    weights[weights < 0] = 0

    img = weights*img/(cp.abs(img) + np.spacing(1))
    del weights

    padding = 64
    padShape = np.pad(img[0], padding, 'symmetric').shape
    imgPad = cp.zeros((len(img), padShape[0], padShape[1]), dtype=np.complex64)
    if np.any(padding) > 0:
        for i in range(len(img)):
            imgPad[i] = cp.pad(img[i], padding, 'symmetric')
            # there is a function for smoothing the edges that I am
            # skipping for now

    dX, dY = getPhaseGradient(imgPad)  # quick
    phase = np.real(getImgInt2D(dX, dY))  # slow

    # remove padding
    idx1 = np.arange(padding, phase.shape[1] - padding, dtype=int)
    # Had to shift here, not too sure why.
    idx2 = np.arange(padding - 1, phase.shape[2] - padding - 1, dtype=int)
    phase = phase[:, idx1][:, :, idx2]

    return phase
