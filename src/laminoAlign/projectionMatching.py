import matplotlib
import matplotlib.pyplot as plt
from laminoAlign import utils, plotting
from laminoAlign.utils import timerStart, timerEnd
import laminoAlign as lam
import numpy as np
import cupy as cp
import scipy
import cupyx.scipy.fft as cufft
import astra
import scipy.io as sio
import time
from IPython.display import clear_output
import traceback
import os


class projectionMatching:
    """
    Projection matching object -- iteratively align the sinogram based 
    on the filtered difference between the model sinogram (forward 
    projection of the reconstruction) and the aligned sinogram.

    Parameters
    ----------
    sinogram : numpy.ndarray
        The projections to be aligned
    weights : numpy.ndarray
        The weights indicating the region of the sinogram that is used 
        in alignment calculations
    binning : int
        The factor by which the sinogram and weights will be 
        downsampled
    updatedGeometry : dict
        The scan geometry parameters that are used in back- and 
        forward-projection
    initialShift : numpy.ndarray
        The shift that will be applied to the sinogram before the 
        projection matching loop

    Attributes
    ----------
    shift : ndarray
        The shift update that is calculated and applied to the sinogram 
        and weights on each iteration of the projection matching loop
    shiftTotal : ndarray
        The total shift applied to the sinogram (equal to initialShift 
        plus all updates from the projection matching loop)
    rec : ndarray
        The back-projection of the sinogram; the 3D reconstruction
    modelSinogram : ndarray
        The forward-projection of the 3D reconstruction rec
    """

    def __init__(self, sinogram=None, weights=None, angles=None, binning=None, Npix=None,
                 config=None, emptyObject=False, refineGeometryOn=False, updatedGeometry={}):
        if not emptyObject:
            self.sinogram = sinogram
            self.weights = weights
            self.angles = angles
            self.binning = binning
            self.refineGeometryOn = refineGeometryOn

            # Get settings from the config file
            # Projection matching settings
            projMatchSettings = ['maxIter', 'highPassFilter', 'stepRelaxation',
                                 'minStepSize', 'localTV', 'localTV_lambda']
            for setting in projMatchSettings:
                setattr(self, setting, config['Projection Matching'][setting])
            # GPU settings
            self.n = config['GPU']['blockLength']

            self.Nangles = self.sinogram.shape[0]
            self.binnedVolume = (len(self.sinogram),
                                 int(np.round(self.sinogram.shape[1]/self.binning)),
                                 int(np.round(self.sinogram.shape[2]/self.binning)))
            self.Npix = np.array(np.ceil(Npix/self.binning), dtype=int)
            print("Binned Sinogram Dimensions:", self.binnedVolume, "pixels")
            print("Reconstruction Dimensions:", self.Npix, "pixels")

            # Get geometry values from updated geometry
            for geometry in updatedGeometry.keys():
                setattr(self, geometry, updatedGeometry[geometry])
            self.CoROffset = self.CoROffset/self.binning

            self.exceptionCaught = False

    def align(self, initialShift):
        """
        Wrapper for the projection-matching alignment sequence. 
        """

        try:
            self.alignTomoConsistencyLinear(initialShift)
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
            self.exceptionCaught = True
        finally:
            lam.utils.freeAllBlocks()
            # Move all GPU arrays back onto the CPU
            for key in self.__dict__.keys():
                if type(self.__dict__[key]) == cp.ndarray:
                    try:
                        self.__dict__[key] = self.__dict__[key].get()
                    except:
                        print("Unable to move attribute", key, "from GPU to CPU")
            # Clear all 3D data objects
            astra.data3d.clear()
            # Free memory
            lam.utils.freeAllBlocks()

    def alignTomoConsistencyLinear(self, initialShift):
        """
        This is the main projection-matching alignment sequence.
        This function iteratively minimizes the cost function that is 
        the filtered difference between the sinogram and the forward-
        projections of the 3D reconstruction.

        Parameters
        ----------
        initialShift : numpy.ndarray
            The shift to be applied to `sinogram` before the alignment 
            loop begins.
        """

        # Number of frames to be used in each GPU calculation chunk
        n = self.n

        # Initialize the object for printing updates
        printer = progressPrinter()

        self.timerOn = False
        tStart = timerStart()

        initialShift = initialShift.astype(np.float32)
        # Save the initial shift
        self.initialShift = initialShift*1
        # shiftTotal keeps track of the total shift applied to the sinogram
        self.shiftTotal = initialShift*1

        if not (hasattr(self, 'inputSinogram') and hasattr(self, 'inputWeights')):
            self.inputSinogram = cp.zeros(self.binnedVolume, dtype=cp.float32)
            self.inputWeights = cp.zeros(self.binnedVolume, dtype=cp.float32)
            self.weights = self.weights/255

            # Downsample and shift the input arrays in GPU-digestible chunks
            Niter = int(np.ceil(self.Nangles/n))
            for i in range(Niter):
                idx = np.arange(n*i, n*(i+1), dtype=int)
                idx = idx[idx < self.Nangles]

                t0 = timerStart()
                self.inputSinogram[idx] = utils.imShiftGeneric(
                    cp.array(self.sinogram[idx]),
                    shift=self.initialShift[idx],
                    downsample=self.binning,
                    interpMethod='fft',
                    interpSign=-1)
                tSino = time.time() - t0

                t0 = timerStart()
                self.inputWeights[idx] = utils.imShiftLinear(
                    cp.array(self.weights[idx]),
                    shift=self.initialShift[idx],
                    downsample=self.binning,
                    method='linear')
                tWeights = time.time() - t0

                print("{:.4f}".format(tSino), "s sinogram downsampling")
                print("{:.4f}".format(tWeights), "s weights downsampling")
                print("Downsampling arrays, iteration", str(i)+"/"+str(Niter))
                clear_output(wait=True)

            # Initialize the arrays that will be shifted on each loop iteration
            self.sinogram = self.inputSinogram*1
            self.weights = self.inputWeights*1
        else:
            # This case is typically only relevant when actively developing the code
            # If pre-downsampled and pre-shifted arrays were included in the object
            # initialization, then just use those
            self.inputSinogram = cp.array(self.inputSinogram)
            self.inputWeights = cp.array(self.inputWeights)
            self.sinogram = self.inputSinogram*1
            self.weights = self.inputWeights*1

        initTime = time.time() - tStart
        timerEnd(tStart, "Initialization Time", self.timerOn)

        print("{:.4f}".format(initTime), "s ", "Initialization Time")

        # Avoid edge issues by multiplying weights by a tukey window
        A = 0.2
        tukeyWindow = cp.array(
            scipy.signal.windows.tukey(self.sinogram.shape[1], A)[:, np.newaxis]
            * scipy.signal.windows.tukey(self.sinogram.shape[2], A), dtype=np.float32)

        # Make a circular mask that will be multiplied with the reconstruction
        circulo = utils.apply3DApodization(np.zeros(self.Npix), 0, 0, 5)

        # Initialize some variables that will be updated in the loop
        # Array of pixel shifts for each sinogram angle:
        self.shift = np.zeros((self.Nangles, 2))
        # All calculated shifts will be stored here:
        self.shiftAll = np.zeros((self.maxIter, self.Nangles, 2))
        self.astraConfig = {}
        self.geometries = {}
        self.tLoops = []

        # Projection matching loop
        for ii in range(self.maxIter):
            self.iteration = ii
            timerEnd(initTime, "Initialization Time\n", self.timerOn, subtract=False)

            tLoop = timerStart()
            if ii > 0:
                shiftUpdate = (self.shiftTotal-self.initialShift)/self.binning
                Niter = int(np.ceil(self.Nangles/n))
                t0 = timerStart()
                for i in range(Niter):
                    idx = np.arange(n*i, n*(i+1), dtype=int)
                    idx = idx[idx < self.Nangles]

                    # Shift the sinogram
                    self.sinogram[idx] = utils.fftImShift(
                        self.inputSinogram[idx],
                        shiftUpdate[idx])
                    # Shift the weights
                    self.weights[idx] = utils.imShiftLinear(
                        self.inputWeights[idx],
                        shiftUpdate[idx],
                        'linear')

                # apply filter to avoid edge issues
                self.weights = self.weights*tukeyWindow

                timerEnd(t0, "Shift Sinogram and Weights", self.timerOn)

            if ii == 0:
                self.MASS = np.median(np.abs(self.sinogram).mean(axis=(1, 2)))

            # Get reconstruction using FBP
            cfg, vectors = lam.astra.initializeAstra(self.sinogram, self.angles, self.Npix,
                                                     self.laminoAngle,
                                                     self.tiltAngle,
                                                     self.skewAngle,
                                                     CoROffset=self.CoROffset)
            self.rec, self.astraConfig, self.geometries = lam.FBP.FBP(self.sinogram,
                                                                      self.weights,
                                                                      cfg, vectors,
                                                                      self.geometries,
                                                                      self.astraConfig,
                                                                      self.timerOn)
            t0 = timerStart()
            self.rec = self.rec*circulo
            timerEnd(t0, "Apply Circular Mask", self.timerOn)

            # Apply TV regularization to the 3D reconstruction
            if self.localTV:
                # Not tested yet!!
                t0 = timerStart()
                self.rec = lam.regularization.chambolleLocalTV3D(self.rec, self.localTV_lambda, 10)
                timerEnd(t0, "TV Minimization", self.timerOn)

            # Calculate the shift for this iteration
            if ii < self.maxIter-1:

                # Estimate the (x,y) shift that minimizes the cost function
                t0 = timerStart()
                self.findOptimalShift(n)
                timerEnd(t0, "findOptimalShift", self.timerOn)

                # Multiply by the step relaxation factor and do some other simple post-processing
                self.postProcessShift()

                # Check if the step size update is small enough to stop the loop
                maxUpdate = np.max(np.quantile(np.abs(self.shift*self.binning), 0.995, axis=0))
                if maxUpdate < self.minStepSize and ii > 0:
                    print("Max step size update: " + "{:.4f}".format(maxUpdate) + " px")
                    print("Minimum step size reached, stopping loop...")
                    break

                # Update the estimates for the scan geometry angles
                if self.refineGeometryOn:
                    self.refineGeometry()

                # Update the total position shift
                self.shiftTotal = self.shiftTotal + self.shift*self.binning
                # Save the shift updates in an array
                self.shiftAll[ii, :, :] = self.shift*self.binning

                # Update the array tracking the execution time of each loop
                self.tLoops = self.tLoops + [time.time()-tLoop]

            timerEnd(tLoop, "TOTAL LOOP TIME", self.timerOn)
            printer.update(ii, self, maxUpdate)

        timerEnd(tStart, "Projection Matching Execution Time", True)

    def findOptimalShift(self, n, timerOn=False):
        """
        Estimate the optimal shift between the sinogram and the 
        sinogram model/re-projections that minimizes the weighted 
        difference:
        || W * (sinogramModel - sinogram + alpha * d(sino)/dX )^2 ||

        Parameters
        ----------
        n : int
            Number of projections used in each GPU calculation chunk
        """

        t0 = timerStart()
        # Get model sinogram by forward-projecting the computed tomogram
        self.sinogramModel = astra.create_sino3d_gpu(
            self.rec,
            self.geometries['proj_geom'],
            self.geometries['vol_geom'])[1].transpose([1, 0, 2])
        timerEnd(t0, "findOptimalShift: Retrieve Sinogram Reconstruction", self.timerOn)
        Niter = int(np.ceil(self.Nangles/n))
        t0 = timerStart()
        for i in range(Niter):
            idx = np.arange(n*i, n*(i+1), dtype=int)
            idx = idx[idx < self.Nangles]
            sinogramModelChunk = cp.array(self.sinogramModel[idx], dtype=np.float32)

            # Get the high-pass filtered residual sinogram
            residSino = sinogramModelChunk - self.sinogram[idx]
            residSino = utils.imFilterHighPass1D(residSino, 2, self.highPassFilter)
            residSino = utils.imFilterHighPass1D(residSino, 2, self.highPassFilter)

            # Calculate the alignment shift
            dX = self.getImgGradFiltered(sinogramModelChunk, 0, self.highPassFilter)
            dX = utils.imFilterHighPass1D(dX, 2, self.highPassFilter)
            xShift = -(np.sum(self.weights[idx]*dX*residSino, axis=(1, 2))
                       / np.sum(self.weights[idx]*dX**2, axis=(1, 2)))
            del dX

            dY = self.getImgGradFiltered(sinogramModelChunk, 1, self.highPassFilter)
            dY = utils.imFilterHighPass1D(dY, 1, self.highPassFilter)
            yShift = -(np.sum(self.weights[idx]*dY*residSino, axis=(1, 2))
                       / np.sum(self.weights[idx]*dY**2, axis=(1, 2)))
            del dY

            if i == 0:
                self.shift = np.array([xShift.get(), yShift.get()]).transpose()
                self.err = (
                    np.sqrt(np.mean((self.weights[idx]*residSino)**2, axis=(1, 2)))/self.MASS
                ).get()
            else:
                self.shift = np.append(
                    self.shift,
                    np.array([xShift.get(), yShift.get()]).transpose(),
                    axis=0)
                self.err = np.append(
                    self.err,
                    (np.sqrt(np.mean((self.weights[idx]*residSino)**2, axis=(1, 2)))/self.MASS
                     ).get())
        timerEnd(t0, "findOptimalShift: Calculate Alignment Shift", self.timerOn)

    def postProcessShift(self):
        # Reduce the shift on this increment by a relaxation factor
        idx = np.abs(self.shift) > 0.5
        self.shift[idx] = 0.5*np.sign(self.shift[idx])
        self.shift = self.shift*self.stepRelaxation

        # Center the shifts around zero
        self.shift[:, 1] = self.shift[:, 1] - np.median(self.shift[:, 1])

        # prevent outliers when the code decides to quickly oscilate around the solution
        maxStep = np.quantile(np.abs(self.shift), 0.99, axis=0)
        maxStep[maxStep > 0.5] = 0.5

        # Do not allow more than 0.5px per iteration
        idx = np.abs(self.shift) > 0.5
        self.shift[idx] = np.min(maxStep)*np.sign(self.shift[idx])

        # Remove degree of freedom in the vertical dimension (avoid drifts)
        orthbase = np.array([np.sin(self.angles*np.pi/180),
                             np.cos(self.angles*np.pi/180)]).transpose()
        A = np.matmul(orthbase.transpose(), orthbase)
        B = np.matmul(orthbase.transpose(), self.shift[:, 0])
        coefs = np.matmul(np.linalg.inv(A), B[:, np.newaxis])

        # Avoid object drifts within the reconstructed FOV
        rigidShift = np.matmul(orthbase, coefs)
        self.shift[:, 0] = self.shift[:, 0] - rigidShift[:, 0]

    def refineGeometry(self, n=20, stepRelaxation=0.01):
        offsetSinoModels = {}
        delta = 0.01
        deltas = [-delta, delta]
        # Get sinogram model at +/- a delta on the laminography angle
        # Get 'infinitesmal' difference of model sinogram with respect to the sinogram angle
        for i in range(2):
            cfg, vectors = lam.astra.initializeAstra(self.sinogram, self.angles, self.Npix,
                                                     self.laminoAngle + deltas[i],
                                                     CoROffset=self.CoROffset)
            geometries = lam.astra.getGeometries(cfg, vectors)
            offsetSinoModels[i] = astra.create_sino3d_gpu(
                self.rec,
                geometries['proj_geom'],
                geometries['vol_geom'])[1].transpose([1, 0, 2])
        dSino = (offsetSinoModels[0]-offsetSinoModels[1])/(2*delta)

        def getGDUpdate(dX, residual, weights, filter):
            dX = lam.utils.imFilterHighPass1D(dX, 2, filter)
            optimalShift = (cp.sum(weights[idx]*residual*dX, axis=(1, 2)) /
                            cp.sum(weights[idx]*dX**2, axis=(1, 2)))
            return optimalShift

        def appendShift(appendShift, shift, iteration):
            if iteration == 0:
                appendShift = shift
            else:
                appendShift = np.append(appendShift, shift)
            return appendShift

        angleStrings = ['laminoAngle', 'tiltAngle', 'skewAngle']
        optimalShift = dict.fromkeys(angleStrings)
        Niter = int(np.ceil(self.Nangles/n))
        for i in range(Niter):
            idx = np.arange(n*i, n*(i+1), dtype=int)
            idx = idx[idx < self.Nangles]

            sinogramModelChunk = cp.array(self.sinogramModel[idx], dtype=np.float32)
            residSinoChunk = sinogramModelChunk - self.sinogram[idx]
            residSinoChunk = utils.imFilterHighPass1D(residSinoChunk, 2, self.highPassFilter)
            dX, dY = lam.math.getImgGrad(sinogramModelChunk)
            dSinoChunk = cp.array(dSino[idx], dtype=np.float32)
            # Get lamino angle correction
            shift = getGDUpdate(dSinoChunk,
                                residSinoChunk,
                                self.weights[idx],
                                self.highPassFilter).get()
            optimalShift['laminoAngle'] = appendShift(optimalShift['laminoAngle'], shift, i)
            optimalShift['laminoAngle'] = appendShift(optimalShift['laminoAngle'], 0, i)
            # Get tilt angle correction
            Dvec = (dX*cp.linspace(-1, 1, dX.shape[1])[:, np.newaxis]
                    - dY*cp.linspace(-1, 1, dY.shape[2]))
            shift = getGDUpdate(Dvec,
                                residSinoChunk,
                                self.weights[idx],
                                self.highPassFilter).get()*180/np.pi
            optimalShift['tiltAngle'] = appendShift(optimalShift['tiltAngle'], shift, i)
            # Get skew angle correction
            Dvec = dY*cp.linspace(-1, 1, dY.shape[2])
            shift = getGDUpdate(Dvec,
                                residSinoChunk,
                                self.weights[idx],
                                self.highPassFilter).get()*180/np.pi
            optimalShift['skewAngle'] = appendShift(optimalShift['skewAngle'], shift, i)

        # Initialize
        if self.iteration == 0:
            self.plotGeometryUpdates = {}
            for k in angleStrings:
                self.plotGeometryUpdates[k] = getattr(self, k)
        # Update results
        for k in optimalShift.keys():
            newValue = getattr(self, k) + np.median(np.real(optimalShift[k]))*stepRelaxation
            setattr(self, k, newValue)
            self.plotGeometryUpdates[k] = appendShift(
                self.plotGeometryUpdates[k],
                newValue, self.iteration)

    def plotRefinedGeometry(self):
        fig, ax = plt.subplots(3, 1)
        fig.set_figheight(6)
        i = 0
        for k in self.plotGeometryUpdates:
            ax[i].plot(self.plotGeometryUpdates[k], '.-')
            ax[i].set_xlabel('Iteration')
            ax[i].set_ylabel('Angle (degrees)')
            ax[i].set_title(k + " update")
            ax[i].grid()
            i = i + 1
        plt.tight_layout()
        plt.show()

    @staticmethod
    def getImgGradFiltered(img, axis, highPassFilter):
        """Get filtered image gradient"""

        xp = cp.get_array_module(img)
        if xp.__name__ == 'numpy':
            backend = 'scipy'
        else:
            backend = cufft
        isReal = img.dtype != np.complexfloating
        Np = img.shape

        if axis == 0:
            X = 2j*xp.pi*scipy.fft.fftshift(xp.arange(0, Np[2], dtype=np.float32)/Np[2] - 0.5)
            with scipy.fft.set_backend(backend):
                img = scipy.fft.fft(img, axis=2)
            img = img*X
            img = utils.imFilterHighPass1D(img, 2, highPassFilter, False)
            with scipy.fft.set_backend(backend):
                img = scipy.fft.ifft(img, axis=2)
        if axis == 1:
            X = 2j*xp.pi*scipy.fft.fftshift(xp.arange(0, Np[1], dtype=np.float32)/Np[1] - 0.5)
            with scipy.fft.set_backend(cufft):
                img = scipy.fft.fft2(img)
            img = img*X[:, xp.newaxis]
            img = utils.imFilterHighPass1D(img, 2, highPassFilter, False)
            with scipy.fft.set_backend(cufft):
                img = scipy.fft.ifft2(img)

        if isReal:
            img = xp.real(img)

        return img

    def saveMovies(self, folder, movieFormat='mp4',
                   quality={'sinogram': 100, 'reconstruction': 500},
                   whichPlots={}, sortByAngle=True):
        """Save movies and GIFs of arrays and shifts used in the 
        projection matching loop

        Parameters
        ----------
        folder : string
            Folder where movies will be saved
        movieFormat : string
            Format that array movies will be saved as -- can be .mp4 
            or .avi
        quality : dict
            Quality (in dots per inch DPI) that each type of array 
            movie will be saved as
        whichPlots : dict
            Specifies which movies will be saved
        sortByAngle : Boolean
            Specify if the movies should be sorted by angle (True) or 
            sorted by scan order (False)
        """

        # Save current backend so you can revert to it at the end
        mpBackend = matplotlib.get_backend()
        matplotlib.use('agg')

        # Default dict specifying which plots to save
        defaultWhichPlots = {'incremental shifts 0': False, 'incremental shifts 1': False,
                             'cumulative shifts 0': False, 'cumulative shifts 1': False,
                             'reconstruction': False, 'aligned sinogram': False,
                             'model sinogram': False, 'initial sinogram': False,
                             'weighted initial sinogram': False}
        if whichPlots == {}:
            whichPlots = defaultWhichPlots
        else:
            for key in defaultWhichPlots:
                if not any([key == newKey for newKey in whichPlots]):
                    whichPlots[key] = False

        # Set default movie parameters for frames per second and dots per inch
        names = ['sinogram', 'reconstruction', 'lineplot']
        defaultParams = {}
        for i in names:
            defaultParams[i] = {}
        defaultParams['lineplot']['DPI'] = 100
        defaultParams['lineplot']['FPS'] = 10
        defaultParams['sinogram']['FPS'] = 25
        defaultParams['reconstruction']['FPS'] = int(np.ceil(self.rec.shape[0]/10))
        # Update parameter quality with user settings
        defaultParams['sinogram']['DPI'] = quality['sinogram']
        defaultParams['reconstruction']['DPI'] = quality['reconstruction']

        plt.rcParams['image.cmap'] = 'bone'

        # Force the plot of calculated shifts to always have roughly 50 frames
        spacing = int(np.ceil(self.shiftAll[:, :, 0].shape[0]/50))

        if sortByAngle:
            idxSort = np.argsort(self.angles)
        else:
            idxSort = np.arange(0, len(self.angles), dtype=int)

        print("Saving movies to " + folder + "...")

        # Save gifs of incremental and cumulative alignment shift at each iteration
        if whichPlots['incremental shifts 0']:
            plotting.animatePlot(
                self.shiftAll[::spacing, idxSort, 0], title="Shift on Iteration",
                xlabel="", ylabel="Shift (Pixels)",
                saveGIF=True, filename=os.path.join(folder, "Shift per Iteration - Direction 0"), 
                FPS=10, frameSkip=spacing)

        if whichPlots['incremental shifts 1']:
            plotting.animatePlot(
                self.shiftAll[::spacing, idxSort, 1], title="Shift on Iteration",
                xlabel="", ylabel="Shift (Pixels)",
                saveGIF=True, filename=os.path.join(folder, "Shift per Iteration - Direction 1"),
                FPS=10, frameSkip=spacing)

        if whichPlots['cumulative shifts 0']:
            plotting.animatePlot(
                (self.shiftAll.cumsum(axis=0)[::spacing, idxSort, 0]
                 + self.initialShift[idxSort, 0]),
                title="Cumulative Shift on Iteration",
                xlabel="", ylabel="Shift (Pixels)", saveGIF=True,
                filename=os.path.join(folder, "Cumulative Shift - Direction 0"), 
                FPS=10, frameSkip=spacing)

        if whichPlots['cumulative shifts 1']:
            plotting.animatePlot(
                (self.shiftAll.cumsum(axis=0)[::spacing, idxSort, 1]
                 + self.initialShift[idxSort, 1]),
                title="Cumulative Shift on Iteration",
                xlabel="", ylabel="Shift (Pixels)", saveGIF=True,
                filename=os.path.join(folder, "Cumulative Shift - Direction 1"), 
                FPS=10, frameSkip=spacing)

        # Save movies of 3D reconstruction, model (or re-projected) sinogram, and various sinograms
        if whichPlots['reconstruction']:
            spc = int(np.ceil(self.rec.shape[0]/10))
            clim = [np.quantile(self.rec[::spc], 0.00001),
                    np.quantile(self.rec[::spc], 0.99999)]
            titleString = "Reconstruction"
            plotting.animateStack(self.rec,
                                  title=titleString,
                                  filename=os.path.join(folder, titleString),
                                  FPS=defaultParams['reconstruction']['FPS'],
                                  DPI=defaultParams['reconstruction']['DPI'],
                                  plotType=movieFormat,
                                  clim=clim)

        if whichPlots['model sinogram']:
            clim = [np.quantile(self.sinogramModel[::20], 0.00001),
                    np.quantile(self.sinogramModel[::20], 0.99999)]
            titleString = "Sinogram from FP of 3D Reconstruction"
            plotting.animateStack(self.sinogramModel[idxSort],
                                  title=titleString,
                                  filename=os.path.join(folder, titleString),
                                  FPS=defaultParams['sinogram']['FPS'],
                                  DPI=defaultParams['sinogram']['DPI'],
                                  plotType=movieFormat,
                                  clim=clim)

        if whichPlots['aligned sinogram']:
            clim = [np.quantile(self.sinogram[::20], 0.00001),
                    np.quantile(self.sinogram[::20], 0.99999)]
            titleString = "Aligned Sinogram"
            plotting.animateStack(self.sinogram[idxSort],
                                  title=titleString,
                                  filename=os.path.join(folder, titleString),
                                  FPS=defaultParams['sinogram']['FPS'],
                                  DPI=defaultParams['sinogram']['DPI'],
                                  plotType=movieFormat,
                                  clim=clim)

        if whichPlots['initial sinogram']:
            titleString = "Initial Sinogram"
            clim = [np.quantile(self.inputSinogram[::20], 0.00001),
                    np.quantile(self.inputSinogram[::20], 0.99999)]
            plotting.animateStack(self.inputSinogram[idxSort],
                                  title=titleString,
                                  filename=os.path.join(folder, titleString),
                                  FPS=defaultParams['sinogram']['FPS'],
                                  DPI=defaultParams['sinogram']['DPI'],
                                  plotType=movieFormat,
                                  clim=clim)

        if whichPlots['weighted initial sinogram']:
            titleString = "Weighted Initial Sinogram"
            clim = [np.quantile(self.inputSinogram[::20], 0.00001),
                    np.quantile(self.inputSinogram[::20], 0.99999)]
            plotting.animateStack(self.inputSinogram[idxSort]*self.inputWeights[idxSort],
                                  title=titleString,
                                  filename=os.path.join(folder, titleString),
                                  FPS=defaultParams['sinogram']['FPS'],
                                  DPI=defaultParams['sinogram']['DPI'],
                                  plotType=movieFormat,
                                  clim=clim)

        # Change backend back to what it was before (inline)
        # This doesn't actually work the way I want it to. 
        # You still have to change it back using the magic command 
        # %matplotlib inline in the notebook.
        matplotlib.use(mpBackend)

    def saveAsDict(self, filename="", folder=""):
        saveDict = {}
        for key in self.__dict__.keys():
            if type(self.__dict__[key]) == cp.ndarray:
                saveDict[key] = self.__dict__[key].get()
            else:
                saveDict[key] = self.__dict__[key]
        sio.savemat(folder + filename, saveDict)


class progressPrinter:
    """Class designed for printing updates in the projection-matching 
    loop

    Attributes
    ----------
        updateFrequency : int
            How often to update the printout
        stepSizeListLength : int
            How many step sizes to list at once in the printout
    """

    def __init__(self, updateFrequency=4, stepSizeListLength=5):
        self.updateFrequency = updateFrequency
        self.stepSizeListLength = stepSizeListLength
        self.shiftUpdateString = ""
        self.ctr = 0
        self.shiftUpdateString = ""

    def update(self, iteration, projMatch, maxUpdate):

        if iteration % self.updateFrequency == 0:
            projMatch.timerOn = True
            clear_output(wait=True)
            self.ctr = self.ctr + 1
            if self.ctr % self.stepSizeListLength == 0:
                self.shiftUpdateString = (
                    "Iteration " + str(iteration)
                    + ": Max step size update: " + "{:.4f}".format(maxUpdate) + " px\n")
            else:
                self.shiftUpdateString = (
                    self.shiftUpdateString
                    + "Iteration " + str(iteration)
                    + ": Max step size update: " + "{:.4f}".format(maxUpdate) + " px\n")
            print(self.shiftUpdateString)
            # Plot the shift
            if iteration > 5:
                idxSort = np.argsort(projMatch.angles)
                fig, ax = plt.subplots(2, 1)
                fig.suptitle("Iteration " + str(iteration))
                ax[0].set_title("Updated Shift")
                ax[0].plot(projMatch.shiftTotal[idxSort])
                ax[1].set_title("Change from Initial Shift")
                ax[1].plot(projMatch.shiftTotal[idxSort] - projMatch.initialShift[idxSort])
                for i in range(2):
                    ax[i].set_xlim([0, projMatch.Nangles])
                fig.set_tight_layout(True)
                plt.show()
            if projMatch.refineGeometryOn:
                projMatch.plotRefinedGeometry()
        else:
            projMatch.timerOn = False
