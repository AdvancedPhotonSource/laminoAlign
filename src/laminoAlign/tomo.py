import cupy as cp
import numpy as np
import laminoAlign as lam
import scipy
import cv2
import cupyx as cpx
import skimage
import astra
import cupyx.scipy.fft as cufft
import scipy.fft
import traceback
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import multiprocessing as mp
import itertools


def unwrap2DfftSplit(img, emptyRegion, weights, ROI={}, polyfitOrder=1, Niter=10):

    try:
        t0 = lam.utils.timerStart()
        if ROI != {}:
            weights = weights[:, ROI[0]][:, :, ROI[1]]
            img = img[:, ROI[0]][:, :, ROI[1]]
        Npix = np.array(np.shape(img))
        weights = cp.array(weights)
        weights = weights.astype(np.float32)/255
        if not np.all(weights.shape[1:3] == Npix[1:3]):
            weights = lam.utils.interpLinear(weights, sizeOut=Npix[1:3])
        weights[weights < 0] = 0
        weights = weights/weights.max()
        lam.utils.timerEnd(t0, timerOn=False)

        phaseBlock = 0
        for i in range(Niter):
            if i == 0:
                imgResid = img
            else:
                imgResid = img*cp.exp(-1j*phaseBlock)
            phaseBlock = phaseBlock + weights*lam.math.unwrap2Dfft(imgResid, weights)
            if emptyRegion != []:
                phaseBlock = lam.utils.removeSinogramRamp(phaseBlock, emptyRegion, polyfitOrder)
        return phaseBlock
    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        traceback.print_exc()


def estimateReliabilityRegion(complexProjection, probeSize, subsample=8):
    "simple reliability estimation based on amplitude of the reconstruction"
    # estimate_reliability_region

    Npix = np.array(complexProjection.shape)
    NpixNew = np.ceil(Npix[1:3]/subsample)
    # Solve the problem in low resolution
    weightSino = lam.utils.interpolateFT(complexProjection, NpixNew)
    probeSize = np.round(probeSize*NpixNew/Npix[1:3])
    weightSino = np.abs(weightSino)
    weightSino = weightSino > 0.1*np.quantile(weightSino, 0.9)
    Y, X = np.meshgrid(
        np.arange(-np.ceil(probeSize[0]/2), np.floor(probeSize[0]/2 + 1)),
        np.arange(-np.ceil(probeSize[1]/2), np.floor(probeSize[1]/2 + 1)),
    )
    probe = (X/probeSize[0]*2)**2 + (Y/probeSize[1]*2)**2 < 1
    kernel = probe/np.sum(probe)
    weightSino = scipy.signal.fftconvolve(weightSino, kernel[np.newaxis], 'same')
    weightSino = weightSino > 0.97  # I think I changed this value
    weightSino = list(weightSino.astype(np.uint8))
    for i in range(len(weightSino)):
        weightSino[i] = (cv2.GaussianBlur(np.float32(1)*weightSino[i], ksize=(0, 0), sigmaX=1,
                                          borderType=cv2.BORDER_DEFAULT)*255).astype(np.uint8)
        weightSino[i] = scipy.ndimage.zoom(weightSino[i], subsample)
        print("iteration " + str(i+1) + "/" + str(len(weightSino)))
        clear_output(wait=True)
    weightSino = np.array(weightSino)/255

    return weightSino


def estimateReliabilityRegion_grad(input, rClose, rErode, unsharp, fill, plotting=False):
    """Use flood-fill to get a mask for the actual object region in each projection"""
    # Still pretty slow!

    downsample = 1

    if downsample > 1:
        Np = np.array(input.shape[1:])
        NpNew = np.round(np.ceil(Np/downsample/2)*2).astype(int)
        n = 20
        Nangles = input.shape[0]
        numBlocks = np.arange(0, np.ceil(Nangles/n), dtype='int')
        inputNew = np.zeros((Nangles, NpNew[0], NpNew[1]))
        for i in numBlocks:
            idx = np.arange(n*i, n*(i+1), dtype=int)
            idx = idx[idx < Nangles]
            inputNew[idx] = lam.utils.interpFT_centered(cp.array(input[idx]), NpNew, -1).get()
        input = inputNew
        print(input.shape)
        rClose = rClose/downsample
        rErode = rErode/downsample

    weights = np.zeros(input.shape)

    # input = np.angle(input) # slow
    closeStructure = cp.array(skimage.morphology.diamond(rClose/downsample))
    erodeStructure = cp.array(skimage.morphology.diamond(rErode/downsample))

    unsharpStructure = cp.array(
        [[-0.1667,   -0.6667,   -0.1667],
         [-0.6667,    4.3333,   -0.6667],
         [-0.1667,   -0.6667,   -0.1667]])

    for i in range(len(input)):
        t0 = lam.utils.timerStart()
        if plotting:
            fig, ax = plt.subplots(2, 3)
            fig.tight_layout()
        tempSino = cp.angle(cp.array(input[i]))
        lam.utils.timerEnd(t0, "Copy to GPU", True)

        if unsharp:
            tempSino = cpx.scipy.ndimage.correlate(tempSino, unsharpStructure)

        sobelx = cpx.scipy.ndimage.sobel(tempSino, 1)
        sobely = cpx.scipy.ndimage.sobel(tempSino, 0)
        tempSino = cp.sqrt(sobelx**2 + sobely**2)
        tempSino[tempSino > 1] = 0

        if plotting:
            r, c = 0, 0
            ax[r, c].imshow(tempSino.get())
            ax[r, c].set_title("sobel")

        ctrPt = (round(tempSino.shape[0]/2), round(tempSino.shape[1]/2))
        tempSino = skimage.segmentation.flood_fill(tempSino.get(), ctrPt, 1)
        if plotting:
            r, c = 0, 1
            ax[r, c].imshow(tempSino)
            ax[r, c].set_title("flood fill")

        level = skimage.filters.threshold_otsu(tempSino)
        tempSino = tempSino > level
        if plotting:
            r, c = 0, 2
            ax[r, c].imshow(tempSino)
            ax[r, c].set_title("otsu threshold")

        if fill > 0:
            # can be put on GPU, but much slower for some reason
            tempSino = scipy.ndimage.binary_fill_holes(tempSino)
        if plotting:
            r, c = 1, 0
            ax[r, c].imshow(tempSino)
            ax[r, c].set_title("binary fill")

        tempSino = cpx.scipy.ndimage.binary_closing(cp.array(tempSino), closeStructure)
        if plotting:
            r, c = 1, 1
            ax[r, c].imshow(tempSino.get())
            ax[r, c].set_title("binary closed")

        tempSino = cpx.scipy.ndimage.binary_erosion(cp.array(tempSino), erodeStructure).get()

        if plotting:
            r, c = 1, 2
            ax[r, c].imshow(tempSino)
            ax[r, c].set_title("binary erode")
        plt.show()

        weights[i] = tempSino

        clear_output(wait=True)
        lam.utils.timerEnd(t0, "Loop Time", True)
        print("iteration " + str(i + 1) + "/" + str(len(input)))

    return weights


def shrinkWeights(weights, rErode, erodeStructure='disk'):

    if erodeStructure == 'disk':
        erodeStructure = cp.array(skimage.morphology.disk(rErode))
    elif erodeStructure == 'diamond':
        erodeStructure = cp.array(skimage.morphology.diamond(rErode))
    else:
        print("Erode structure not properly specified -- aborting process")
        return

    dtype = weights.dtype
    for i in range(len(weights)):
        weights[i] = cpx.scipy.ndimage.binary_erosion(cp.array(weights[i]),
                                                      erodeStructure).get()
    weights = weights.astype(dtype)

    return weights


def rotPoints(x, y, theta):
    theta = theta*np.pi/180
    xr = np.cos(theta)*x + np.sin(theta)*y
    yr = -np.sin(theta)*x + np.cos(theta)*y
    return xr, yr


def processPoints(x, y, pixelSize, rot=0):
    "For rotating scan points in `getScanPositions"
    # Scan points need to be rotated and recentered
    # x, y, and pixelSize should be in microns

    # micronToPixel = 1e-6/pixelSize
    micronToPixel = 1/pixelSize
    xr, yr = rotPoints(x, y, rot)
    xr = xr-xr.min()
    xr = xr*micronToPixel
    yr = yr-yr.min()
    yr = yr*micronToPixel
    return xr, yr


def getMask(positions, Npix, probe, pixelSize, sigmaX=24, offset=0, useGPU=False, device=0):

    try:
        if useGPU:
            xp = cp
            cp.cuda.Device(device).use()
            t0 = time.time()
            probe = xp.array(probe)
        else:
            xp = np

        # Get the probe illumination at each of the scan positions
        xr, yr = processPoints(positions[:, 0], positions[:, 1], pixelSize)
        xr = xr + offset
        yr = yr + offset

        positions = np.round(np.array([xr, yr]).transpose())
        fullArray = xp.zeros(Npix, dtype=np.float32)

        mask = lam.utils.addToProjection(probe.transpose(), fullArray, positions)
        if useGPU:
            mask = mask.get()
        mask = mask.astype(np.float32)

        # Turn it into a mask
        def applyThresh(mask, thresh):
            idx = mask > thresh
            mask[~idx] = 0
            mask[idx] = 1
        for i in range(1):
            mask = cv2.GaussianBlur(
                mask,
                ksize=(0, 0),
                sigmaX=sigmaX,
                borderType=cv2.BORDER_DEFAULT)
            threshVal = 0.5
            thresh = np.quantile(mask, threshVal)
            applyThresh(mask, thresh)

            return mask

    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        print(traceback.format_exc())
    finally:
        lam.utils.freeAllBlocks()


def getWeightsFromPositions(positions, probe, stack, pixelSize, blurSigmaX=24,
                            offset=0, useGPU=True, numProcesses=int(mp.cpu_count()*0.8)):
    """Get the weights/mask for each projection by using the measured scan positions and probe"""

    probe = np.abs(probe)**2
    try:
        pool = mp.pool.ThreadPool(numProcesses)
        masks = []
        masks = pool.starmap(getMask,
                             zip(positions,
                                 [proj.shape for proj in stack],
                                 itertools.repeat(probe),
                                 itertools.repeat(pixelSize),
                                 itertools.repeat(blurSigmaX),
                                 itertools.repeat(offset),
                                 itertools.repeat(useGPU),
                                 itertools.repeat(cp.cuda.get_device_id())))
    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        print(traceback.format_exc())
    finally:
        lam.utils.freeAllBlocks()
        pool.close()  # prevent more processes from being submitted
        pool.join()  # wait for processes to end
        pool.terminate()
        del pool

    return masks


def phaseRampRemoval(objectFull, weights, theta, Npix, binning, Niter, CoR, blockLength,
                     laminoAngle, tiltAngle, skewAngle):

    shift = np.zeros((len(theta), 2))
    objectROI = {}
    objectROI[0] = np.arange(0, objectFull.shape[1], dtype=int)
    objectROI[1] = np.arange(0, objectFull.shape[2], dtype=int)
    shift = np.zeros((len(theta), 2))
    Npix = np.ceil(Npix/binning).astype(int)
    CoR = CoR/binning

    binnedVolume = (len(objectFull),
                    int(objectFull.shape[1]/binning),
                    int(objectFull.shape[2]/binning))
    binnedObject = cp.zeros(binnedVolume, dtype=np.complex64)
    binnedWeights = cp.zeros(binnedVolume, dtype=np.float32)

    n = blockLength
    N = int(np.ceil(len(theta)/n))

    # Downsample complex projections
    for i in range(N):
        idx = np.arange(n*i, n*(i+1), dtype=int)
        idx = idx[idx < len(theta)]
        binnedObject[idx] = lam.utils.imShiftGeneric(
            cp.array(objectFull[idx]),
            shift=shift,
            ROI=objectROI,
            downsample=binning,
            interpMethod='fft',
            interpSign=-1)
    lam.utils.freeAllBlocks()

    # Downsample weights
    for i in range(N):
        idx = np.arange(n*i, n*(i+1), dtype=int)
        idx = idx[idx < len(theta)]
        binnedWeights[idx] = lam.utils.imShiftGeneric(
            cp.array(weights[idx]),
            shift=shift,
            ROI=objectROI,
            downsample=binning,
            interpMethod='fft',
            interpSign=-1)
    lam.utils.freeAllBlocks()

    gammaTot = 1
    xGammaTot = 0
    yGammaTot = 0
    circulo = lam.utils.apply3DApodization(np.zeros(Npix.astype(int)), 0, 0, 10)

    for i in range(Niter):

        astraConfig = {}
        geometries = {}
        phase = lam.math.unwrap2Dfft(cp.array(binnedObject), cp.array(binnedWeights))

        cfg, vectors = lam.astra.initializeAstra(phase, theta, Npix, laminoAngle,
                                                 tiltAngle, skewAngle, CoR)
        rec, astraConfig, geometries = lam.FBP.FBP(phase, binnedWeights, cfg, vectors,
                                                   geometries, astraConfig)
        del phase
        rec = -rec*circulo

        proj = astra.create_sino3d_gpu(rec,
                                       geometries['proj_geom'],
                                       geometries['vol_geom'])[1].transpose([1, 0, 2])

        # Avoid edge effects
        binnedWeights[:, [0, -1]] = 0

        imgRef = cp.exp(-1j * cp.array(proj) * (1 - binnedWeights))
        with scipy.fft.set_backend(cufft):
            binnedObject, gamma, xGamma, yGamma = lam.utils.stabilizePhase(
                binnedObject, binnedWeights, imgRef, normalizeAmplitude=False)

        gammaTot = gammaTot * gamma
        xGammaTot = xGammaTot + xGamma
        yGammaTot = yGammaTot + yGamma

    # Calculate amplitude correction
    aobject = np.abs(binnedObject)
    ampCorrection = cp.median(aobject, axis=(1, 2)).reshape(len(binnedObject), 1, 1)

    def applyRampShifted(objectFull, gamma, xGamma, yGamma, ampCorrection):
        xp = cp.get_array_module(objectFull)
        gamma = xp.array(gamma)
        xGamma = xp.array(xGamma)
        yGamma = xp.array(yGamma)
        ampCorrection = xp.array(ampCorrection)

        [M, N] = xp.shape(objectFull)[1:3]
        xRamp = xp.pi*xp.linspace(-1, 1, M, dtype=np.float32).reshape(M, 1)
        yRamp = xp.pi*xp.linspace(-1, 1, N, dtype=np.float32).reshape(1, N)
        objectFull = objectFull*(gamma/ampCorrection)
        objectFull = objectFull*xp.exp(1j*xRamp*xGamma)
        objectFull = objectFull*xp.exp(1j*yRamp*yGamma)
        return objectFull

    objectFull = applyRampShifted(
        objectFull, gamma.get(), xGamma.get(), yGamma.get(), ampCorrection.get())

    return objectFull


def findImgRotation2D(img, maxRange=[-22.5, 22.5]):
    """Find the rotation of the object that yields the most sparse 
    features"""

    def sparseness(x):
        order1 = 1
        order2 = 2
        x = x[:]
        sqrt_n = np.sqrt(len(x))
        spars = ((sqrt_n - np.linalg.norm(x, order1) / np.linalg.norm(x, order2))/(sqrt_n - order1))

        return spars

    def getScore(data, angle):
        Npix = np.shape(data)
        [X, Y] = cp.meshgrid(cp.arange(-np.ceil(Npix[1]/2), np.floor(Npix[1]/2), dtype=np.float32),
                             cp.arange(-np.ceil(Npix[0]/2), np.floor(Npix[0]/2), dtype=np.float32))
        data = data*(X**2/(Npix[1]/2)**2 + Y**2/(Npix[0]/2)**2 < 1/2)
        data = data - cpx.scipy.ndimage.gaussian_filter(data, 5)

        with scipy.fft.set_backend(cufft):
            data = lam.utils.fftRotate(data, angle)
            data = data[np.ceil(data.shape[0]*0.1):np.floor(data.shape[0]*0.9),
                        np.ceil(data.shape[1]*0.1):np.floor(data.shape[1]*0.9)]
            data = np.abs(scipy.fft.fftshift(scipy.fft.fft2(data)))
        score = -np.mean([sparseness(data.mean(axis=0)).get(),
                          sparseness(data.mean(axis=1)).get()])
        return score

    # Grid search first to avoid local minimums
    testImg = cp.abs(img)
    testImg = testImg - cp.median(testImg)
    testImg[testImg < 0] = 0

    def findMin(img, alphaRange):
        N = len(alphaRange)
        score = np.zeros((N, 1))
        for i in range(N):
            score[i] = getScore(img, alphaRange[i])
        return score

    N = 500
    alphaRange = np.linspace(maxRange[0], maxRange[-1], N)
    nextRange = np.array([-1, 1])
    for i in range(3):
        score = findMin(testImg, alphaRange)
        plt.plot(alphaRange, score[:, 0])
        alphaRange = alphaRange[np.argmin(
            score)] + np.arange(nextRange[0], nextRange[1], nextRange[1]/10)
        nextRange = nextRange/10
    angle = alphaRange[np.argmin(score)]
    plt.show()

    return angle


def getRotationAngles(rec):

    Lh = int(rec.shape[1]/2)
    rotAngle = np.zeros(3)
    rotAngle[0] = findImgRotation2D(cp.squeeze(cp.array(rec[:, Lh, :])))
    rotAngle[1] = findImgRotation2D(cp.squeeze(cp.array(rec[:, :, Lh])))
    rotAngle[2] = findImgRotation2D(cp.array(rec.mean(axis=0)))

    return rotAngle


def applyMaskToTomogram(tomogram, circulo, n=50):
    """Apply a circular mask to the tomogram"""

    numBlocks = np.arange(0, np.ceil(len(tomogram)/n), dtype='int')
    for i in numBlocks:
        tomogram[i*n:(i+1)*n] = (cp.array(tomogram[i*n:(i+1)*n])*circulo).get()
    return tomogram
