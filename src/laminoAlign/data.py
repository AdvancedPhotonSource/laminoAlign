import laminoAlign as lam
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
import skimage
import matplotlib.pyplot as plt
import time as time
import os
import copy
import scipy.io as sio
import pickle
import h5py as h5
import traceback
from pathlib import Path
import configparser
import astra
from IPython.display import clear_output


class data:
    """The `data` class contains data, parameters, and wrapper methods 
    needed for alignment and reconstruction of laminography datasets.

    Attributes
    ----------
    stackObject : ndarray
        The complex projections that will be processed, aligned, and 
        back-projected in order to produce a 3D reconstruction.
    angles : ndarray
        The rotation angle at which each projection was measured.
    sinogram : ndarray
        The result of doing a phase unwrapping of the stackObject. 
    weightSino : ndarray
        The weights indicating the region of the region of the 
        stackObject and sinogram used in various calculations such as 
        phase unwrapping, back-projection, and calculating the 
        alignment shifts.
    updatedGeometry : dict
        The scan geometry parameters that are used in back and forward-
        projection of the sinogram.
    config : dict
        The `config` attribute contains the settings that will be used 
        by default by methods of `data`.

    Notes
    -----
    There are two ways to initialize a data object:
        1) Load raw data: `data = laminoAngle.data.data(config, 
        fileReader)`
        2) Load a saved data object: `data = laminoAngle.data.data(
            reload=True, loadFolder=loadFolder)`
    To generate a saved data object that can be reloaded, use the 
    method `data.saveAll(filepath)` to save the data object
    """

    def __init__(self, config=None, fileReader=None, reload=False, loadFolder=None,
                 loadStackObject=False, timerOn=True):
        
        # Turn off FFT caching -- necessary for GPU memory management
        cp.fft.config.get_plan_cache().set_size(0)

        if not reload:
            # Load angles
            self.angles, removedScanNums, scanNums = fileReader.loadAngles()

            # Sometimes scan numbers are duplicated and have multiple
            # angle measurements associated with a single scan number. 
            # I remove these scans because I don't know which angle 
            # measurement is correct.
            if len(removedScanNums) > 0:
                fileReader.projFilepaths = {k: fileReader.projFilepaths[k] for k in scanNums}
                fileReader.scansTomo = scanNums

            # Load projections
            self.stackObject = fileReader.parLoadStackObject(fileReader.scansTomo)
            self.scansTomo = fileReader.scansTomo*1
            if len(self.stackObject) == len([]):
                return

            # Get parameters p from first projection
            [self.probe, self.object0, self.p] = fileReader.loadPtychoReconstruction(
                fileReader.projFilepaths[fileReader.scansTomo[0]])

            # Combine probe into single image if necessary
            if len(self.probe.shape) == 3:
                self.probe = self.probe.sum(axis=0)

            # Load the settings from the config file
            self.assignConfig(config)

            # Create the dict for storing updated geometry parameters
            self.updatedGeometry = {
                'laminoAngle': self.config['Scan Geometry']['laminoAngle'],
                'tiltAngle': self.config['Scan Geometry']['tiltAngle'],
                'skewAngle': self.config['Scan Geometry']['skewAngle'],
                'CoROffset': np.array([self.config['Projection Matching']['offsetCoR_offset'],
                                       self.config['Projection Matching']['offsetCoR_V']])
            }

            # Get index that will be used to make plots that are in 
            # order of ascending angle
            self.idxAngleSort = np.argsort(self.angles)

            # Get scan positions
            try:
                [self.scanPositions,
                 self.scanPositionsSource] = fileReader.getScanPositions()
            except Exception:
                # This code usually doesn't even need scan positions, 
                # so we'll ignore it if it getScanPositions doesn't work
                print("Scan positions not loaded!")
                pass

            self.timerOn = timerOn

        elif reload:
            lam.data.data.reloadData(self, loadFolder, loadStackObject)

    def getWeightsFromPositions(self):
        """Get weights from scan positions. 
        This only works for some formats of positions. It is not 
        advised to use this function unless you know what you're doing 
        and you confirm that the weights produced by this function are 
        reasonable."""

        if self.scanPositionsSource != 'reconstructions':
            print("WARNING: weights not calculated.\n" +
                  "The scan positions are not loaded from the projections' file path, " +
                  "so they cannot be used to calculate weights.")

        self.weightSino = lam.tomo.getWeightsFromPositions(
            self.scanPositions,
            self.probe,
            self.stackObject,
            self.p['dx_spec'].ravel()[0],
            useGPU=False,
            numProcesses=50)

    def equalizeStackSize(self, shape=[], repairOrientation=False, padType='constant'):
        """Convert the stackObject (and weights, if they exist) from 
        list to array.

        Attributes
        ----------
        See `convertStackListToArray` for details."""

        self.stackObject = lam.data.data.convertStackListToArray(self.stackObject,
                                                                 padType=padType)
        self.newShape = np.array(self.stackObject.shape[1:])
        if hasattr(self, 'weightSino'):
            self.weightSino = lam.data.data.convertStackListToArray(self.weightSino,
                                                                    self.newShape,
                                                                    padType=padType)

    @staticmethod
    def convertStackListToArray(stack, shape=None, repairOrientation=False, padType='constant',
                                divisBy=32):
        """Convert list of imagaes into array of images by cropping and
        padding each image so that they each have the same dimensions.

        Parameters
        ----------
        stack : list
            List of 2D arrays to be converted to a 3D array.
        shape : array_like
            Shape of the new array. The size of the largest projection 
            will be chosen if this is left blank.
        padType : str
            Padding method that will be passed to `numpy.pad`.
        divisBy : str
            The width and length of each projections will be forced to 
            be divisble by `2*divisBy`. `divisBy` should be set to the 
            minimum value downsampling value that will be used with 
            other functions.
        """

        # Reorient the projections -- only needed for specific data 
        # sets where projections are 90 degrees off from where they
        # should be
        if repairOrientation:
            print("Rotating and flipping some projections...")
            t0 = lam.utils.timerStart()
            targetAspect = stack[0].shape[1]/stack[0].shape[0]
            for i in range(len(stack)):
                aspectRatio = stack[i].shape[1]/stack[i].shape[0]
                reorientProjection = ((targetAspect < 1 and aspectRatio > 1) or
                                      (targetAspect > 1 and aspectRatio < 1))
                if reorientProjection:
                    print("Reorienting", i)
                    stack[i] = np.fliplr(np.rot90(stack[i], -1))
            lam.utils.timerEnd(t0, "Rotating and flipping some projections...Completed", True)

        if shape is None:
            shape = np.array([proj.shape for proj in stack]).max(axis=0)
        else:
            shape = np.array(shape)
        # Force new shape to be compatible with downsampling functions
        newShape = (np.floor(shape/(divisBy*2))*(divisBy*2)).astype(int)

        # Fix stack dimensions through cropping and padding
        print("Fixing stack dimensions...")
        t0 = lam.utils.timerStart()
        lam.utils.fixStackDims(stack, newShape, padType)
        lam.utils.timerEnd(t0, "Fixing stack dimensions...Completed", True)

        # Convert to array
        print("Converting list to array...")
        t0 = lam.utils.timerStart()
        stack = np.array(stack)
        lam.utils.timerEnd(t0, "Converting list to array..Completed", True)

        return stack

    @staticmethod
    def downsampleStack(stack, binning, shift=None, n=5):
        """
        Downsample along axis = (1, 2) of a 3D array using FFT 
        interpolation. \\
        Suitable for use with sinogram, weightSino, or stackObject.

        Parameters
        ----------
        stack : list
            3D array to be downsampled.
        binning : int
            The factor to downsample by. The stack dimensions should be 
            an even multiple of the binning. 
            Ex: A stack with dimensions (10, 320, 640) can be 
            downsampled by 32 because the dimensions (320, 640) = 
            (10*32, 20*32) but a stack with dimensions (10, 352, 640) 
            cannot be downsampled by 32 because (352, 640) = (11*32, 
            20*32).
        shift : ndarray
            Optional shift to apply to each layer of the stack.
        """

        if shift is None:
            shift = np.ones((len(stack), 2))

        try:
            Niter = int(np.ceil(len(stack)/n))
            Nangles = len(stack)
            newVolume = (Nangles,
                         int(stack.shape[1]/binning),
                         int(stack.shape[2]/binning))
            newStack = np.zeros(newVolume, dtype=stack.dtype)
            for i in range(Niter):
                idx = np.arange(n*i, n*(i+1), dtype=int)
                idx = idx[idx < Nangles]
                newStack[idx] = lam.utils.imShiftGeneric(cp.array(stack[idx]),
                                                         downsample=binning,
                                                         interpMethod='fft',
                                                         shift=shift[idx],
                                                         interpSign=-1).get()
                print(str(i) + '/' + str(Niter))
                clear_output(wait=True)
            return newStack
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

    def useDownsampledStack(self, newStack):
        """Replace the `stackObject` with a new downsampled stack and 
        update attributes related to the resolution and array size"""

        M = self.stackObject.shape[1]/newStack.shape[1]
        self.p['dx_spec'] = self.p['dx_spec']*M
        self.stackObject = newStack

    def assignConfig(self, config):
        """Take a config and convert it into a dict attribute of the 
        data object

        Parameters
        ----------
        config : configparser.ConfigParser
            `ConfigParser` object that contains the default parameters 
            for some of the wrapper methods in `data`.
        """

        self.config = {}
        includeSections = config.sections()
        specialCases = [['Projection Matching', 'binningArray']]
        for i in includeSections:
            self.config[i] = {}
            for j in config[i].keys():
                if not any([i == case[0] and j == case[1] for case in specialCases]):
                    # Convert the elements of the ConfigParser to the proper types
                    if config[i][j].isnumeric():
                        self.config[i][j] = int(config[i][j])
                    elif config[i][j].lower() == 'true':
                        self.config[i][j] = True
                    elif config[i][j].lower() == 'false':
                        self.config[i][j] = False
                    else:
                        try:
                            self.config[i][j] = float(config[i][j])
                        except Exception as ex:
                            self.config[i][j] = config[i][j]

        # Load data for special cases (like lists)
        self.config['Projection Matching']['binningArray'] = [
            int(i) for i in config['Projection Matching']['binningArray'].split(',')
        ]

    def insertNewConfigs(self, updateConfig={}):
        """Override specific config settings. This is used to allow the
        user to override the settings from the `config` attribute in 
        individual method calls.

        The user has no need to use access function directly.

        Parameters
        ----------
        updateConfig : dict
            The new parameters that the user wants to use. Ex: the dict 
            `{'Scan Geometry': {'laminoAngle': 42}}` updates the 
            parameter 'laminoAngle' in section 'Scan Geometry'to 42.
        """

        if updateConfig != {}:
            config = lam.data.data.overrideConfigs(self.config, updateConfig)
        else:
            config = copy.deepcopy(self.config)
        return config

    @staticmethod
    def overrideConfigs(config, configUpdate={}):
        """
        Function wrapped by `insertNewConfigs`.

        Parameters
        ----------
        config : dict
            The configuration from data.config.
        configUpdate : dict
            Dict with the same structure as `config` containing the params 
            you want to update.

        Returns
        ----------
        newConfig : dict
            Dictionary with updated settings.
        """

        newConfig = copy.deepcopy(config)
        for subsection in configUpdate.keys():
            for key in config[subsection].keys():
                if any([key == newKey for newKey in configUpdate[subsection]]):
                    newConfig[subsection][key] = configUpdate[subsection][key]
        return newConfig

    def removeScans(self, scanList):
        """Remove individual projections

        Parameters
        ----------
        scanList : list
            List of scan numbers to remove from all present arrays. 
            This should be entered in terms of the scan number as 
            listed in the `scansTomo` attribute.
        """

        # Get indices of projections to be removed
        idx = np.setdiff1d(
            np.arange(0, len(self.scansTomo)),
            [np.where(self.scansTomo == scan)[0][0] for scan in scanList]
        )

        self.angles = self.angles[idx]
        self.scansTomo = self.scansTomo[idx]
        self.idxAngleSort = np.argsort(self.angles)

        stacks = ['stackObject', 'sinogram', 'weightSino']
        for stackString in stacks:
            if hasattr(self, stackString):
                setattr(self, stackString, getattr(self, stackString)[idx])

        # Update the reconstruction volume
        if hasattr(self, 'NpixAlign'):
            lam.data.data.reconstructionPrep()

    def removeTiltAndSkew(self, updateConfig={}):
        """Remove the tilt and skew on the projections by the amount 
        specified in the `updatedGeometry` attribute"""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        t0 = lam.utils.timerStart()

        self.stackObject = lam.data.data.affineTransform(
            self.stackObject,
            -self.updatedGeometry['tiltAngle'],
            -self.updatedGeometry['skewAngle'],
            n=config['GPU']['blockLength'])

        if hasattr(self, 'weightSino'):
            # If the weights are uint8 this will not work properly
            dtypeWeights = self.weightSino.dtype

            self.weightSino = lam.data.data.affineTransform(
                self.weightSino,
                -self.updatedGeometry['tiltAngle'],
                -self.updatedGeometry['skewAngle'],
                rotationType=2,
                n=config['GPU']['blockLength'])

            # Convert weights to uint8
            for i in range(len(self.weightSino)):
                idx = self.weightSino[i] > 0.5
                self.weightSino[i][idx] = 1
                self.weightSino[i][~idx] = 0
                # Convert to uint8, this is for compatibility with the 
                # projection matching function
                self.weightSino[i] = (self.weightSino[i]*255).astype(np.uint8)

        self.updatedGeometry['tiltAngle'] = 0
        self.updatedGeometry['skewAngle'] = 0

        lam.utils.timerEnd(t0, "Affine Transform Time", self.timerOn)

    @staticmethod
    def affineTransform(images, tiltAngle, skewAngle, rotationType=1, n=20):
        # This *should* be an in-place method, but the behavior is
        # a bit off. Be careful with how you use this, and use the
        # other cases in this file as a template for how it should
        # be used.

        nDims = len(images.shape)
        if nDims == 3:
            iterations = np.arange(0, np.ceil(len(images)/n), dtype='int')
        else:
            iterations = [0]
            images = images[np.newaxis]

        # If rotation is more than +/- 45 degrees, an initial 90
        # degree rotation must be applied because the dimensions of the
        # new image will be reversed
        images = lam.utils.rotateStackMod90(images, tiltAngle)
        try:
            with scipy.fft.set_backend(cufft):
                for i in iterations:
                    idx = np.arange(n*i, n*(i+1), dtype=int)
                    idx = idx[idx < len(images)]

                    # I accidentally mixed views and fancy indexing 
                    # here, could this be what is causing the 
                    # unexpected behavior with regards to in-place 
                    # processing?
                    images[idx] = lam.utils.initialImageProcess(
                        cp.array(images[i*n:(i+1)*n]), 
                        tiltAngle,
                        skewAngle,
                        rotationType).get()

                    clear_output(wait=True)
                    print("iteration " + str(i + 1) + "/" + str(iterations[-1] + 1))
            if nDims != 3:
                images = images[0]
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

        return images

    @staticmethod
    def plotSamples(imageStack, imageFunc=None, N=5, vlim=None, ROI=None):
        """Plot a few frames and some information"""

        if imageFunc is None:
            def imageFunc(x): return x

        print("Single Frame Size:", imageStack[0].shape[0:],
              "\nNumber of Frames:", len(imageStack))

        fig, ax = plt.subplots(1, N)
        fig.set_figwidth(10)
        for i in range(N):
            idx = np.linspace(0, len(imageStack) - 1, N, dtype=int)[i]
            if ROI is not None:
                plotImage = imageFunc(imageStack[idx][ROI[0]][:, ROI[1]])
            else:
                plotImage = imageFunc(imageStack[idx])
            if vlim is not None:
                ax[i].imshow(plotImage, vmin=vlim[0], vmax=vlim[1])
            else:
                ax[i].imshow(plotImage)
            ax[i].axis('off')
            ax[i].set_title("Frame " + str(idx))
        plt.show()

    def plotShift(self, shifts, title="", labels=None, sortShifts=True, figSize=(10, 5),
                  colors=plt.rcParams['axes.prop_cycle'].by_key()['color']):
        """Method for quick-plotting of one or more shift arrays.

        Parameters
        ----------
        shifts : list
            List of arrays to plot
        """

        # Plot the result
        fig, ax = plt.subplots()
        fig.set_figwidth(figSize[0])
        fig.set_figheight(figSize[1])
        ctr = 0
        for shift in shifts:
            if len(shift.shape) == 1:
                shift = shift[:, np.newaxis]
            for i in range(shift.shape[1]):
                if sortShifts:
                    x = self.angles[self.idxAngleSort]
                    y = shift[self.idxAngleSort, i]
                    linestyle = '-'
                else:
                    x = np.arange(0, len(self.angles))
                    y = shift[:, i]
                    linestyle = '-'
                plt.plot(x, y, marker='.', linestyle=linestyle, markersize=3, linewidth=1.4,
                         color=colors[ctr])
                ctr = ctr+1
        if labels is not None:
            plt.legend(labels)
        plt.xlim([x[0], x[-1]])
        plt.grid('on')
        plt.ylabel('Shift (px)')
        plt.xlabel('Angle (deg)')
        plt.title(title)

    @staticmethod
    def notebookAnimate(stack, imageFunc=None, vlim=[], xlim=[], ylim=[], title=""):
        """Animate a 3D array in the Jupyter notebook"""

        if imageFunc is None:
            def imageFunc(x): return x

        for i in range(len(stack)):
            if len(vlim) == 0:
                plt.imshow(imageFunc(stack[i]))
            else:
                plt.imshow(imageFunc(stack[i]), vmin=vlim[0], vmax=vlim[1])
            if len(xlim) != 0:
                plt.xlim(xlim)
            if len(ylim) != 0:
                plt.ylim(ylim)
            plt.title(title + " " + str(i))
            plt.show()
            clear_output(wait=True)

    def showCoR(self, image, CoROffset=[], w=[], ax=""):
        """
        Plot an image with a dot where the center of rotation is.
        The input image must be centered properly!
        """

        if len(CoROffset) == 0:
            CoROffset = self.updatedGeometry['CoROffset']

        lam.plotting.plotImage(image, widths=w, ax=ax)
        CoR = np.array(image.shape)/2 + CoROffset[::-1]
        plt.plot(CoR[1], CoR[0], '.m', label='Estimated Center of Rotation')
        print("Center of Rotation:", CoR)
        plt.legend()
        return plt

    def setGeneralROI(self, initialShape, widths=[], centers=[], divisBy=32):
        """General method for getting ROI. Initial shape is the size of 
        each projection. The ROI is forced to be some even multiple of 
        divisBy which is necessary for compatibility with FFT 
        downsampling functions.
        
        If you don't enter widths or centers, the default is the entire 
        object range.
        """

        if len(widths) == 0:
            widths = np.array(initialShape)
        if len(centers) == 0:
            centers = (np.array(initialShape)/2).astype(int)
        # Width must be an even multiple of divisBy
        widths = np.floor(np.array(widths)/(divisBy*2))*(divisBy*2)
        ROI = {}
        ROI[0] = np.arange(-widths[0]/2, widths[0]/2, dtype=int) + int(np.round(centers[0]))
        ROI[1] = np.arange(-widths[1]/2, widths[1]/2, dtype=int) + int(np.round(centers[1]))
        isBadROI = (np.any(ROI[0] < 0)
                    or np.any(ROI[1] < 0)
                    or np.any(ROI[0] > initialShape[0])
                    or np.any(ROI[1] > initialShape[1]))
        if isBadROI:
            print("Range of interest extends outside of stackObject shape.",
                  "Use different parameters.")
            ROI = {}

        return ROI

    def setObjectROI(self, widths=[], centers=[], divisBy=32):
        """Select the region of the stackObject that will be used in 
        sinogram and weights generation."""
        self.objectROI = lam.data.data.setGeneralROI(self, self.stackObject.shape[1:], widths,
                                                     centers, divisBy)
        print("Object ROI:",
              "\n - Vertical:", self.objectROI[0][0], "to", self.objectROI[0][-1],
              "\n - Horizontal:", self.objectROI[1][0], "to", self.objectROI[1][-1])

    def setXCorrROI(self, widths=[], centers=[], divisBy=32):
        """Select the region of the stackObject that will be used in 
        cross-correlation pre-alignment"""
        self.xCorrROI = lam.data.data.setGeneralROI(self, self.stackObject.shape[1:], widths,
                                                    centers, divisBy)
        print("Cross-correlation ROI:",
              "\n - Vertical:", self.xCorrROI[0][0], "to", self.xCorrROI[0][-1],
              "\n - Horizontal:", self.xCorrROI[1][0], "to", self.xCorrROI[1][-1])

    def setAlignmentROI(self, widths=[], centers=[], divisBy=32):
        """Select the region of the sinogram that will be used in 
        projection-matching alignment and 3D reconstruction."""
        self.alignmentROI = lam.data.data.setGeneralROI(self, self.sinogram.shape[1:], widths,
                                                        centers, divisBy)
        print("Alignment ROI:",
              "\n - Vertical:", self.alignmentROI[0][0], "to", self.alignmentROI[0][-1],
              "\n - Horizontal:", self.alignmentROI[1][0], "to", self.alignmentROI[1][-1])

    def getIllumSum(self, skipAffine=False):
        """Estimate the variable `illumSum`: the illumination intensity 
        for different regions of the object.
        
        `illumSum` is used in `crossCorrelationAlign` and 
        `removePhaseRamp`"""

        if 'positions' in self.p:
            objSize = self.p['object_size'][:, 0]
            positionOffset = np.round(self.p['positions'])
            fullArray = np.zeros(objSize)
            smallArray = np.abs(self.probe)**2
            self.illumSum = lam.utils.addToProjection(smallArray, fullArray, positionOffset)
            self.illumSum = lam.utils.cropPad(
                self.illumSum[np.newaxis],
                self.newShape[0],
                self.newShape[1])[0]
        else:
            objectSize = self.newShape
            probeSize = np.array(self.probe.shape[0:2])
            self.illumSum = np.ones(objectSize - probeSize)
            self.illumSum = lam.utils.cropPad(
                self.illumSum[np.newaxis],
                objectSize[0],
                objectSize[1])[0]
        # semi-normalize
        self.illumSum = self.illumSum/np.quantile(self.illumSum, 0.9)
        if not skipAffine:
            # Affine transform the illumSum variable to make consistent 
            # with stackObject
            try:
                # pass
                self.illumSum = lam.data.data.affineTransform(
                    self.illumSum,
                    -self.config['Scan Geometry']['tiltAngle'],
                    -self.config['Scan Geometry']['skewAngle'])
            except Exception as ex:
                print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
                traceback.print_exc()
            finally:
                lam.utils.freeAllBlocks()

        self.illumSum = self.illumSum.astype(np.float32)

    def removePhaseRamp(self, n=5):
        """Stabilize the phase of the complex projections `stackObject` 
        to be near zero and remove the linear ramp and phase offset"""

        t0 = lam.utils.timerStart()
        try:
            numBlocks = np.arange(0, np.ceil(len(self.stackObject)/n), dtype='int')
            for i in numBlocks:
                idx = np.arange(n*i, n*(i+1), dtype=int)
                idx = idx[idx < len(self.angles)]
                with scipy.fft.set_backend(cufft):
                    self.stackObject[idx] = lam.utils.stabilizePhase(
                        cp.array(self.stackObject[idx]),
                        cp.array(self.illumSum/self.illumSum.max()))[0].get()
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()
        lam.utils.timerEnd(t0, "Phase Ramp Removal")

    def crossCorrelationAlign(self, updateConfig={}):
        """Perform coarse alignment using the cross-correlation 
        method"""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        try:
            # Get shifts from cross-correlation alignment
            self.xCorrShift, self.variation, self.variationAligned, self.xCorrCumShifts = (
                lam.coarseAlign.alignTomoXCorr(self.stackObject,
                                               self.illumSum,
                                               self.angles,
                                               self.xCorrROI,
                                               config['GPU']['blockLength'],
                                               config['Cross Correlation Alignment']))
            # Clear uneeded variables
            delattr(self, 'variation')
            delattr(self, 'variationAligned')
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

    def estimateReliabilityRegion(self, method=1, updateConfig={}):
        """Calculate the region of the `stackObject` that is good 
        enough to be unwrapped. This method essentally just highlights 
        the actual object region.

        Wrapper function for `laminoAlign.tomo` methods.

        Parameters:
        ----------
        method : int
            Determines which phase unwrapping method to use. Can be set 
            to 1 or 2. Method 1 should generally be used because 2 
            usually does not work.
        updateConfig : dict
            The dictionary containing values that will override the 
            settings in `self.config`. The relevant settings are the 
            values in the `Reliability Region` section. Play with the 
            `rClose` and `rErode` settings, which are used in 
            morphological operations of
            `laminoAlign.tomo.estimateReliabilityRegion_grad`

        New Attributes:
        ----------
        weightSino : ndarray
            Mask that specifies the regions of the `stackObject` that 
            will be used in phase unwrapping. This region will also be 
            used in projection matching alignment calculations.
        """

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        # Change to views later
        if (len(self.objectROI[0]) == self.stackObject.shape[1]
                and len(self.objectROI[1]) == self.stackObject.shape[2]):
            stackObjectCropped = self.stackObject
        else:
            stackObjectCropped = self.stackObject[:, self.objectROI[0]][:, :, self.objectROI[1]]

        try:
            if method == 1:
                rClose = config['Reliability Region']['rClose']
                rErode = config['Reliability Region']['rErode']
                unsharp = config['Reliability Region']['unsharp']
                fill = config['Reliability Region']['fill']
                self.weightSino = lam.tomo.estimateReliabilityRegion_grad(
                    stackObjectCropped,
                    rClose,
                    rErode,
                    unsharp,
                    fill)
            elif method == 2:
                self.weightSino = lam.tomo.estimateReliabilityRegion(
                    stackObjectCropped,
                    self.p['asize'][0:2, 0])
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()
        self.weightSino = self.weightSino.astype(np.float32)

    def unwrapPhase2D(self, updateConfig={}):
        """Apply 2D phase unwrapping to the complex projection 
        `stackObject`.

        Wrapper function for `laminoAlign.tomo.unwrap2DfftSplit`."""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        Nangles = len(self.stackObject)
        sinogramShape = (Nangles, len(self.objectROI[0]), len(self.objectROI[1]))
        self.sinogram = np.zeros(sinogramShape, dtype=np.float32)

        useROI = not (len(self.objectROI[0]) == self.stackObject.shape[1] and
                      len(self.objectROI[1]) == self.stackObject.shape[2])

        # airGap = [widthSinogram/4, widthSinogram/4]
        # Fine for now, but might need to be changed for different 
        # types of samples
        airGap = []  

        n = config['GPU']['blockLength']
        numBlocks = np.arange(0, np.ceil(Nangles/n), dtype='int')

        # Unwrap the phase of the complex projection `stackObject`
        try:
            for i in numBlocks:
                print("iteration " + str(i+1) + "/" + str(numBlocks[-1]))
                clear_output(wait=True)
                idx = np.arange(n*i, n*(i+1), dtype=int)
                idx = idx[idx < Nangles]
                if useROI:
                    stackChunk = (
                        self.stackObject[idx][:, self.objectROI[0]][:, :, self.objectROI[1]]
                    )
                    weightsChunk = (
                        self.weightSino[idx][:, self.objectROI[0]][:, :, self.objectROI[1]]
                    )
                else:
                    stackChunk = self.stackObject[idx]
                    weightsChunk = self.weightSino[idx]
                with scipy.fft.set_backend(cufft):
                    self.sinogram[idx] = lam.tomo.unwrap2DfftSplit(
                        cp.array(stackChunk),
                        airGap,
                        weightsChunk,
                    ).get()
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

    @staticmethod
    def fftImShiftStack(stack, shift, n=20):
        """Shift a stack of images with sub-pixel precision using a FFT 
        based method.

        Wrapper function for `laminoAlign.utils.fftImShift`

        Parameters:
        ----------
        stack : ndarray
            Three-dimensional stack of images to be shifted.
        shift : ndarray
            Two-dimensional array of shifts to be applied. Should have 
            shape (N,2) where N is the number of images in `stack` and 
            the two columns contain the horizontal and vertical shift.

        Returns:
        ----------
        stack : ndarray
            The shifted stack.
        """

        try:
            N = len(stack)
            Niter = round(np.ceil(N/n))
            for i in range(Niter):
                idx = np.arange(n*i, n*(i+1), dtype=int)
                idx = idx[idx < N]
                stack[idx] = lam.utils.fftImShift(cp.array(stack[idx]), shift[idx]).get()
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

        return stack

    def reconstructionPrep(self, divisBy=32, updateConfig={}):
        """Calculate number of pixels that will be in the 3D 
        reconstruction of the object."""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        downsampleProjections = 0
        # pixelSize = self.p['dx_spec'][0]*2**downsampleProjections
        pixelSize = self.p['dx_spec'].ravel()[0]

        # Calculate number of pixels in the reconstructed object
        NpixAlign = np.ceil(
            0.5*len(self.alignmentROI[1])
            / np.cos(np.pi/180*(self.config['Scan Geometry']['laminoAngle'] - 0.01))
        )
        NpixAlign = np.ceil(
            np.array([NpixAlign, NpixAlign, config['Sample']['sampleThickness']/pixelSize])
            / divisBy)*divisBy
        self.NpixAlign = NpixAlign.astype(int)

        print("Number of pixels in full resolution 3D reconstruction:", self.NpixAlign, "px")
        print("Pixel Size:", "{:.1f}".format(pixelSize*1e9), "nm")
        reconstructionDims = [round(i, 1) for i in self.NpixAlign*pixelSize*1e6]
        print("Reconstruction dimensions:", reconstructionDims, "um")

    def saveShifts(self, projMatch, filename="", folder=""):
        """Save projection matching shift and cross-correlation shift 
        to a .mat file"""

        saveDict = {}
        saveDict['projMatch'] = {}
        saveDict['projMatch']['shiftTotal'] = projMatch.shiftTotal
        saveDict['xCorrShift'] = self.xCorrShift
        sio.savemat(os.path.join(folder, filename), saveDict)

    def runReconstructionSequence(self, folder="", shifts=[], updateConfig={}, saveData='none',
                                  usePreProcessData=False, whichPlots={}, refineGeometryOn=False,
                                  updatedGeometry={}, keepDownsampledArrays=False):
        """Run the projection-matching alignment (PMA) sequence at the 
        resolutions indicated in the config.

        Parameters:
        ----------
        folder : string, optional
            Folder to aligment results to.
        shifts : ndarray, optional
            2D array with shifts to be applied at the beginning of the 
            first PMA loop.
        updateConfig : dict, optional
            Dictionary with overriding settings. Relevant settings are 
            in the `Projection Matching` and `GPU` sections.
        saveData : string, optional
            Indicates when (if at all) to save data. Options are 
            \"none\", \"all\", or \"last\". The data that will be 
            saved are movies of variables used in the PMA loop and a 
            .mat file of the alignment shifts.
                \"none\": No data will be saved
                \"all\": Data will be saved after the PMA loop for all 
                       resolutions
                "last\": Data will be saved after the PMA loop for only 
                        the last resolution            
        whichPlots : dict, optional
            Dictionary indicating what movies to save on each loop. The 
            default is that none will be saved. See 
            `laminoAlign.projectionMatching.saveMovies` for details. 
                Ex: `whichPlots = {'cumulative shifts 0': True, 
                'cumulative shifts 1': True,
                'reconstruction': True, 
                'aligned sinogram': True}` 
            means that the movies of the total alignment shift vs loop 
            iteration number, the reconstruction after the PMA loop, 
            and the aligned sinogram after the PMA loop will be saved.
        updatedGeometry : dict, optional
            Scan geometry values used in forward- and back-projection 
            operations. Defaults to using the values in 
            `self.updatedGeometry`.
        refineGeometryOn : boolean, optional
            Toggles the option for refining the scan geometry angles.
        usePreProcessData: boolean, optional
            Only used for development purposes. Please ignore.
        keepDownsampledArrays: boolean, optional
            Only used for development purposes. Please ignore.

        New Attributes:
        ----------
        projMatch : laminoAlign.projectionMatching.projectionMatching
            `projectionMatching` object with results from the most 
            recent PMA loop.
        """

        config = lam.data.data.insertNewConfigs(self, updateConfig)
        binning = config['Projection Matching']['binningArray']

        if np.shape(shifts) == np.shape([]):
            shifts = np.zeros((len(self.angles), 2))

        if not hasattr(self, 'saveInputs'):
            # Create a dict to save the initial sinograms in.
            # Saving the inputs can save time by eliminating 
            # initialization time. It is useful for methods like 
            # estimateCoROffset where initialization is done many times.
            self.saveInputs = {}

        # Run the PMA loop for all of the specified resolutions
        for i in range(len(binning)):

            # Set boolean for saving results
            if (saveData == 'all') or (saveData == 'last' and i == len(binning) - 1):
                saveThisRound = True
            else:
                saveThisRound = False
            # Generate folder name where results will be saved
            subFolder = ('Binning ' + str(binning[i]) + ' Iterations '
                         + str(config['Projection Matching']['maxIter']))
            self.saveFolder = os.path.join(folder, subFolder, '')

            # Perform the projection-matching alignment at the next 
            # resolution
            self.projMatch = lam.data.data.getReconstruction(
                self, shifts, binning[i], saveThisRound, config, usePreProcessData, whichPlots,
                refineGeometryOn, updatedGeometry)

            # End the sequence if an exception was thrown during
            # projectionMatching.alignTomoConsistencyLinear
            if self.projMatch.exceptionCaught:
                print("Exception caught during alignment, ending reconstruction sequence...")
                return

            print("Alignment at binning", binning[i], "completed...")

            # Use alignment shift result as the initial shift in the 
            # next resolution
            shifts = self.projMatch.shiftTotal

            # Save the initial downsampled arrays that haven't been 
            # shifted
            if keepDownsampledArrays:
                self.saveInputs[binning[i]] = {}
                self.saveInputs[binning[i]]['inputSinogram'] = self.projMatch.inputSinogram
                self.saveInputs[binning[i]]['inputWeights'] = self.projMatch.inputWeights

    def getReconstruction(self, shifts, binning, saveData=False, config={},
                          usePreProcessData=False, whichPlots={}, refineGeometryOn=False,
                          updatedGeometry={}):
        """Start the projection-matching alignment loop at the 
        resolution indicated by `binning`.

        This function is designed to be accessed from 
        `runReconstructionSequence` and is a wrapper for the projection-
        matching alignment method in `laminoAlign.projectionMatching`.

        Parameters:
        ----------
        See `runReconstructionSequence` for details.
        """

        # Update updatedGeometry (which holds the up-to-date info on 
        # scan geometry) if the user specified new values
        if updatedGeometry == {}:  # No update
            updatedGeometry = self.updatedGeometry
        else:  # Update with new values
            for k in self.updatedGeometry.keys():
                if k not in updatedGeometry.keys():
                    updatedGeometry[k] = self.updatedGeometry[k]*1
                    print(updatedGeometry[k] is self.updatedGeometry[k])

        # Crop the input sinogram and weights according to the 
        # alignment ROI
        if (len(self.alignmentROI[0]) == self.sinogram.shape[1]
                and len(self.alignmentROI[1]) == self.sinogram.shape[2]):
            alignmentSinogram = self.sinogram
            alignmentWeights = self.weightSino
        else:
            alignmentSinogram = (
                self.sinogram[:, self.alignmentROI[0][0]:self.alignmentROI[0][-1] + 1,
                              self.alignmentROI[1][0]:self.alignmentROI[1][-1] + 1])
            alignmentWeights = (
                self.weightSino[:, self.alignmentROI[0][0]:self.alignmentROI[0][-1] + 1,
                                self.alignmentROI[1][0]:self.alignmentROI[1][-1] + 1])

        # Initialize the projection matching object
        projMatch = lam.projectionMatching.projectionMatching(
            alignmentSinogram,
            alignmentWeights,
            self.angles + 0.1,
            binning,
            self.NpixAlign,
            config,
            refineGeometryOn=refineGeometryOn,
            updatedGeometry=copy.deepcopy(updatedGeometry)
        )

        # Only relevant when developing/debugging: save the initial 
        # downsampled sinogram in the data object so I can eliminate 
        # initialization time on the next method call
        if usePreProcessData and np.any([k == binning for k in self.saveInputs.keys()]):
            projMatch.inputSinogram = self.saveInputs[binning]['inputSinogram']
            projMatch.inputWeights = self.saveInputs[binning]['inputWeights']

        # Execute the projection-matching alignment loop
        projMatch.align(shifts)
        lam.utils.freeAllBlocks()

        if not projMatch.exceptionCaught:
            if saveData:
                # Save results
                if not os.path.isdir(self.saveFolder):
                    os.makedirs(self.saveFolder)
                lam.data.data.saveShifts(self, projMatch, 'shifts.mat', self.saveFolder)
                # Specify quality (in terms of dots per sq inch DPI) 
                # for saved movies
                quality = {'sinogram': 200, 'reconstruction': 500}
                lam.projectionMatching.projectionMatching.saveMovies(
                    projMatch, self.saveFolder, quality=quality, movieFormat='mp4',
                    whichPlots=whichPlots)

        return projMatch

    def estimateCoROffset(self, updateConfig={}):
        """Estimate the center of rotation (CoR) of the projections by 
        getting the projection-matching alignment (PMA) error as a 
        functon of CoR.

        See an example script for details on how to use this method."""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        hCenter = config['Estimate CoR']['offsetCoR_H_center']
        vCenter = config['Estimate CoR']['offsetCoR_V_center']
        hRange = config['Estimate CoR']['offsetCoR_H_range']
        vRange = config['Estimate CoR']['offsetCoR_V_range']
        hIter = config['Estimate CoR']['offsetCoR_H_iterations']
        vIter = config['Estimate CoR']['offsetCoR_V_iterations']

        # Get the horizontal and vertical offsets to test
        hOffsets = np.linspace(hCenter - hRange/2, hCenter + hRange/2, hIter)
        vOffsets = np.linspace(vCenter - vRange/2, vCenter + vRange/2, vIter)

        # Initialize dict to keep results in
        self.estimatedCoR = {}
        for key in ['err', 'mean err', 'shiftTotal']:
            self.estimatedCoR[key] = {}

        # Make a copy of updatedGeometry that will be updated with a new CoR on each iteration
        updatedGeometry = copy.deepcopy(self.updatedGeometry)

        # Clear the saved inputs if they exist
        self.saveInputs = {}

        # Reformat binning setting if necessary
        if type(config['Estimate CoR']['binning']) is not list:
            config['Estimate CoR']['binning'] = [config['Estimate CoR']['binning']]

        # Get the PMA error as a function of CoR
        for i in range(len(hOffsets)):
            for j in range(len(vOffsets)):
                # Use the PMA settings specified by the Estimate CoR config section
                config['Projection Matching'] = {'offsetCoR_V': vOffsets[j],  # Redundant
                                                 'offsetCoR_offset': hOffsets[i],  # Redundant
                                                 'binningArray': config['Estimate CoR']['binning'],
                                                 'maxIter': config['Estimate CoR']['maxIter']}

                # Update the config with the CoR to test this round
                updatedGeometry['CoROffset'] = np.array([hOffsets[i], vOffsets[j]])

                # Run the PMA sequence
                lam.data.data.runReconstructionSequence(self,
                                                        updateConfig=config,
                                                        usePreProcessData=True,
                                                        updatedGeometry=updatedGeometry,
                                                        keepDownsampledArrays=True)

                # Record the results
                kOffs = (hOffsets[i], vOffsets[j])
                self.estimatedCoR['err'][kOffs] = self.projMatch.err
                self.estimatedCoR['mean err'][kOffs] = np.mean(self.projMatch.err)
                self.estimatedCoR['shiftTotal'][kOffs] = self.projMatch.shiftTotal

                plt.clf()
                # Plot the results of all iterations so far
                fig, ax = plt.subplots(1, 2)
                fig.set_figwidth(13)
                fig.set_figheight(2)
                y = [v for v in self.estimatedCoR['mean err'].values()]
                ax[0].plot([k[0] for k in self.estimatedCoR['mean err'].keys()], y, '.')
                ax[0].set_title("Error vs Horizontal Offset")
                ax[1].plot([k[1] for k in self.estimatedCoR['mean err'].keys()], y, '.')
                ax[1].set_title("Error vs Vertical Offset")
                plt.show()

        # Estimate the best CoR by finding the CoR with the lowest error
        idx = np.argmin([v for v in self.estimatedCoR['mean err'].values()])
        CoROffset = [k for k in self.estimatedCoR['mean err'].keys()][idx]
        print("Estimated Center of Rotation (H,V):",
              (round(CoROffset[0], 1), round(CoROffset[1], 1)))

        delattr(self, 'saveInputs')

        return CoROffset

    def updateCoROffset(self, CoROffset):
        """Permanently change the center of rotation (CoR) in the 
        config file"""

        # Load the existing config.ini file
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(self.configPath)

        # Update the CoR offset in the config file
        config.set('Projection Matching', 'offsetCoR_offset', str(CoROffset[0]))
        config.set('Projection Matching', 'offsetCoR_V', str(CoROffset[1]))

        # Save the changes back to the file
        with open(self.configPath, 'w') as configfile:
            config.write(configfile)

        lam.data.data.assignConfig(self, config)

    def recenterObjects(self, CoROffset, useGPU=False, updateConfig={}):
        """Shift arrays so that the center of rotation is (0, 0) and 
        update `updatedGeometry`"""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        CoROffset = np.array(CoROffset)
        shift = -CoROffset*np.ones((len(self.angles), 2))
        arraysToCenter = ['sinogram', 'weightSino', 'stackObject']

        for arr in arraysToCenter:
            if hasattr(self, arr):
                # Avoid wrapping by cutting off the wrapped part of the weights
                if arr == 'weightSino':
                    cutEdge = True
                else:
                    cutEdge = False

                if not useGPU:
                    # Shift the array
                    setattr(self, arr, lam.utils.imShiftLinear(getattr(self, arr), shift),
                            cutEdge=cutEdge)
                else:
                    try:
                        n = config['GPU']['blockLength']
                        N = len(getattr(self, arr))
                        numBlocks = np.arange(0, np.ceil(N/n), dtype='int')
                        tempArray = getattr(self, arr)

                        # Shift the array
                        for i in numBlocks:
                            idx = np.arange(n*i, n*(i+1), dtype=int)
                            idx = idx[idx < N]
                            clear_output(wait=True)
                            print("Shifting " + arr + "...\nIteration", str(i) + "/"
                                  + str(len(numBlocks)))

                            # Shift the array chunk
                            tempArray[idx] = lam.utils.imShiftLinear(
                                cp.array(tempArray[idx]),
                                shift,
                                cutEdge=cutEdge).get()

                    except Exception as ex:
                        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
                        traceback.print_exc()
                    finally:
                        lam.utils.freeAllBlocks()

        self.updatedGeometry['CoROffset'] = np.array([0, 0])

    def removePhaseRamp_final(self, updateConfig={}):
        """Remove the phase ramp from the complex projections 
        `stackObject`"""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        try:
            self.stackObject = lam.tomo.phaseRampRemoval(
                self.stackObject*1,
                self.weightSino,
                self.angles,
                self.NpixAlign.astype(int),
                config['Phase Ramp Removal']['binning'],
                config['Phase Ramp Removal']['Niter'],
                self.updatedGeometry['CoROffset'],
                config['GPU']['blockLength'],
                self.updatedGeometry['laminoAngle'],
                self.updatedGeometry['tiltAngle'],
                self.updatedGeometry['skewAngle'])
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

    def astraReconstruct(self, updateConfig={}):
        """Generate the 3D reconstruction using the astra toolbox

        New Attributes:
        ----------
        tomogram : ndarray
            3D reconstruction of the object

        Returns:
        ----------
        geometries : dict
            The specific projection and volume geometries for this 
            data/experiment. Intended to be used as an input of the 
            `forwardProject` method.
        """

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        try:
            astraConfig = {}
            geometries = {}

            cfg, vectors = lam.astra.initializeAstra(
                self.sinogram,
                self.angles + 0.1,
                self.NpixAlign.astype(int),
                self.updatedGeometry['laminoAngle'],
                self.updatedGeometry['tiltAngle'],
                self.updatedGeometry['skewAngle'],
                self.updatedGeometry['CoROffset']
            )

            # Calculate the 3D reconstruction
            self.tomogram, astraConfig, geometries = lam.FBP.FBP(
                self.sinogram, self.weightSino, cfg, vectors)

            return geometries
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()
            astra.data3d.clear()

    @staticmethod
    def forwardProject(tomogram, geometries):
        """Get the forward-projection a 3D reconstruction

        Parameters: 
        ----------
        tomogram : ndarray
            3D reconstruction of the object that will be forward-
            projected.
        geometries : dict
            The specific projection and volume geometries for this 
            data/experiment.

        Returns:
        ----------
        sinogramModel : ndarray
            The forward projection of the tomogram.
        """

        sinogramModel = astra.create_sino3d_gpu(
            tomogram,
            geometries['proj_geom'],
            geometries['vol_geom'])[1].transpose([1, 0, 2])

        return sinogramModel

    #### Functions for tomogram rotation ####

    def getRotationAngles(self, tomogram=None):
        """Find the rotation angles that will align the 3D 
        reconstruction along its principal axes"""

        try:
            if tomogram is None:
                tomogram = self.tomogram
            self.tomoRotAngles = lam.tomo.getRotationAngles(tomogram)
            clear_output()
            print("Rotation angles stored in tomoRotAngles:", self.tomoRotAngles)
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

    def rotateTomogram(self, tomogram=None, updateConfig={}):
        """Rotate the tomogram by the values specified in 
        `self.tomoRotAngles`"""

        config = lam.data.data.insertNewConfigs(self, updateConfig)

        if tomogram is None:
            tomogram = self.tomogram

        def rotFunc(stack, n, rotAngle):
            numBlocks = np.arange(0, np.ceil(len(stack)/n), dtype='int')
            for i in numBlocks:
                stack[i*n:(i+1)*n] = lam.utils.fftRotate(
                    cp.array(stack[i*n:(i+1)*n]),
                    rotAngle).get()
            return stack

        try:
            n = config['GPU']['blockLength']
            self.tomogram_rot = tomogram*1
            # Rotate the tomogram
            with scipy.fft.set_backend(cufft):
                # Axis 1
                tmp = np.transpose(self.tomogram_rot, [1, 0, 2])
                tmp = rotFunc(tmp, n, self.tomoRotAngles[0])
                self.tomogram_rot = np.transpose(tmp, [1, 0, 2])

                # Axis 2
                tmp = np.transpose(self.tomogram_rot, [2, 1, 0])
                tmp = rotFunc(tmp, n, -self.tomoRotAngles[1])
                self.tomogram_rot = np.transpose(tmp, [2, 1, 0])

                # Axis 0
                self.tomogram_rot = rotFunc(self.tomogram_rot, n, self.tomoRotAngles[2])

            print("tomogram rotated by", self.tomoRotAngles,
                  "\nRotated tomogram is stored in tomogram_rot")
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            lam.utils.freeAllBlocks()

    def useGeometryRefinementResults(self):
        """Update the values of `updatedGeometry` with the results from 
        the PMA geometry refinement"""

        updateStrings = ['laminoAngle', 'tiltAngle', 'skewAngle']

        print("Old Geometry Values:")
        for key in updateStrings:
            print("\t-", key + ":", "{:.4f}".format(self.updatedGeometry[key]))
        self.updatedGeometry['laminoAngle'] = self.projMatch.laminoAngle
        self.updatedGeometry['tiltAngle'] = self.projMatch.tiltAngle
        self.updatedGeometry['skewAngle'] = self.projMatch.skewAngle
        print("New Values:")
        for key in updateStrings:
            print("\t-", key + ":", "{:.4f}".format(self.updatedGeometry[key]))

    ###### Saving and Loading ######

    def setFolders(self, parentFolder):
        """Create a dict of folders where results will be saved"""

        self.folders = {}
        self.folders['parent'] = parentFolder
        self.folders['data objects'] = os.path.join(parentFolder, 'data objects')
        self.folders['alignment results'] = os.path.join(parentFolder, 'alignment results')


    def saveAll(self, name, saveStackObject=None):
        """Save the data object. The large arrays will be saved to a h5 
        file and other attributes are saved into a .obj file.

        Re-create the data object by using the `reloadData` method.
        """

        if saveStackObject is None:
            saveStackObject = not hasattr(self, 'sinogram')
        saveFolder = os.path.join(self.folders['data objects'], name)
        # Save the data instance, if it exists
        lam.utils.partialPickle(self, saveFolder, 'dataObject.obj', exclude='projMatch')
        lam.utils.saveArrays(self, saveFolder, 'arrays.h5')
        # Save the reconstruction instance, if it exists
        if hasattr(self, 'projMatch'):
            lam.utils.partialPickle(self.projMatch, saveFolder, 'dataObject_reconstruction.obj',
                                    exclude=['saveInputs'])
            lam.utils.saveArrays(self.projMatch, saveFolder, 'arrays_reconstruction.h5')
        print("data object saved to:", saveFolder)

    def reloadData(self, folder, loadStackObject=False):
        """Reload a saved data object.

        Parameters:
        ----------
        folder : string
            The folder that contains the .h5 and .obj files with the 
            data object attributes"""

        self.__dict__ = {}

        def loadInstance(objInst, objFilename, arrayFilename):
            "Load data object instance"

            fileObj = open(os.path.join(folder, objFilename), 'rb')
            dataDict = pickle.load(fileObj)
            fileObj.close()

            for key, value in dataDict.items():
                setattr(objInst, key, value)

            # Load arrays into the data object instance
            filepath = os.path.join(folder, arrayFilename)
            with h5.File(filepath, 'r') as myFile:
                for attr in myFile.keys():
                    if not (attr == 'stackObject' and not loadStackObject):
                        if attr.endswith('_real') or attr.endswith('_imag'):
                            # Special case loading of the complex valued stackObject
                            attr = attr[0:-5]
                            # if not hasattr(objInst, attr):
                            setattr(objInst, attr,
                                    myFile[attr + '_real'][:] + myFile[attr + '_imag'][:]*1j)
                        else:
                            setattr(objInst, attr, myFile[attr][:])

        loadInstance(self, 'dataObject.obj', 'arrays.h5')

        if (Path(os.path.join(folder, 'dataObject_reconstruction.obj')).exists() 
            and Path(os.path.join(folder, 'arrays_reconstruction.h5')).exists()):
            try:
                self.projMatch = lam.projectionMatching.projectionMatching(emptyObject=True)
                loadInstance(self.projMatch, 'dataObject_reconstruction.obj',
                             'arrays_reconstruction.h5')
            finally:
                pass

    def saveAsTIFF(self, stack, filename, lims=None, folder=""):
        """Save a stack of images as a .tiff file. The image is 
        rescaled so that the pixel values cover the uint16 range 
        (from 0 to 65535).

        Returns:
        ----------
        im : uint16 ndarray
            The rescaled image.
        """

        if folder == "":
            folder = self.folders['parent']
        validExtensions = (".tiff", ".tif", ".TIFF", ".TIF")
        if not filename.endswith(validExtensions):
            filename = filename + ".tiff"
        filepath = os.path.join(folder, filename)

        # Convert to uint16 and have the largest dynamic range possible
        if lims is None:
            im = stack - stack.min()
            im = (im*65535/im.max()).astype(np.uint16)
        else:
            im = stack - lims[0]
            idx = im < 0
            im[idx] = 0
            im = (im*65535/(lims[1] - lims[0])).astype(np.uint16)
        print("Saving data to " + filepath)
        skimage.io.imsave(filepath, im)
        print("Data saved to " + filepath)

        return im

    def saveAsVTI(self, stack, filename, folder=""):
        """Save a stack of images in .vti format. Compatible with 
        Paraview and other 3D modelling software."""

        if folder == "":
            folder = self.folders['parent']
        validExtensions = (".vti", ".VTI")
        if not filename.endswith(validExtensions):
            filename = filename + ".vti"
        filepath = os.path.join(folder, filename)

        lam.vtkWriter.saveAsVTK(stack, filepath)

