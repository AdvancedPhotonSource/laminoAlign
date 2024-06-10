# Functions for handling data loading
# The data can come in a lot of different formats, and the multiple
# classes here reflect that.

import numpy as np
import pandas as pd
import re
import h5py
import os
import multiprocessing as mp
import scipy.io as sio
import configparser
import glob
import traceback
import laminoAlign as lam


def getConfig(filepath):
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserves case
    config.read(filepath)
    return config


class fileReader:
    def __init__(self, config):

        self.scansTomo = np.arange(int(config['File Info']['scanStart']),
                                   int(config['File Info']['scanEnd']))
        self.basePath = config['File Info']['basePath']
        self.filePrefix = config['File Info']['filePrefix']
        self.fileSuffix = config['File Info']['fileSuffix']
        self.fileExtension = config['File Info']['fileExtension']
        self.analysisPath = os.path.join(self.basePath, 'analysis')
        self.OmnyAngleFile = os.path.join(self.basePath, config['File Info']['scanNumbersPath'])
        self.OmnyPosFile = os.path.join(self.basePath, config['File Info']['positionsPath'])

        if config['File Info']['removeScans'] != '':
            self.removeScans = [int(i) for i in config['File Info']['removeScans'].split(',')]
        else:
            self.removeScans = []

        self.findProjectionFilepaths()

    def findProjectionFilepaths(self, scanNums=[]):
        """Find filepath for the ptychography reconstruction for the 
        first scan"""

        if scanNums == []:
            scanNums = self.scansTomo

        self.projFilepaths = {}
        for scanNum in scanNums:
            # Find the file folder for the given scan number
            for string in os.listdir(self.analysisPath):
                try:
                    [startScan, endScan] = re.findall('S(\w*)-(\w*)', string)[0]
                    strLength = len(startScan)
                    scanString = 'S'+('{:0' + str(strLength) + 'd}').format(scanNum)
                    if int(startScan) <= scanNum and scanNum <= int(endScan):
                        folder = os.path.join(self.analysisPath + string, scanString)
                        break
                except:
                    pass
            # Find the file to load
            pattern = (scanString + self.filePrefix + '\w*' + self.fileSuffix
                       + '.' + self.fileExtension)
            fileFound = False
            try:
                for file in os.listdir(folder):
                    if re.findall(pattern, file):
                        filepath = folder + file
                        self.projFilepaths[scanNum] = filepath
                        fileFound = True
                        break
            except:
                pass
            finally:
                if fileFound == False:
                    print("No projection data found for scan", scanNum)
                else: 
                    print("Found scan", str(scanNum) + ":", filepath)

    def loadAngles(self):
        """
        Inputs: 
            parameters
            scans: same as .scansTomo
            tomo_id: ID number of sample, empty by default
            plot_angles: plot loaded angles? True by default
        Returns: 
            parameters
            angles: Array of laminography rotation angles
            dupeScans: scans that were duplicated in the angle file
            scanNums: the scan numbers that correspond to each angle
            """

        numScans = len(self.scansTomo)
        angles = np.zeros((1, numScans))
        S = self.readOmnyAngles()

        # Remove angles where data is duplicated
        dupeScans = S.index[S.index.duplicated()]
        S = S.drop(labels=dupeScans, axis='index')
        scanNums = S.index.to_numpy()

        # Add 0.1 to avoid angles being too well aligned with pixels
        # ie avoid exact angles 0, 90, 180
        # Negative because lamni has a different angle direction
        # definition
        angles = -(S['readoutAngle'] + 0.1).to_numpy()

        return angles, dupeScans, scanNums

    def readOmnyAngles(self):

        columns = ['scanNumber', 'targetAngle', 'readoutAngle', 'tomoID', 'subTomoNum',
                   'detPosNum', 'sampleName']
        outmat = pd.read_csv(
            self.OmnyAngleFile, delimiter=' ', header=None, keep_default_na=False,
            names=columns, usecols=columns, index_col='scanNumber')
        keepScans = np.intersect1d(outmat.index, self.scansTomo)
        keepScans = np.setdiff1d(keepScans, self.removeScans)
        outmat = outmat.loc[keepScans]
        # Remove scans if they are not in the omny angle file, or if
        # manually set to remove those scans
        self.scansTomo = keepScans
        self.projFilepaths = {k: self.projFilepaths[k] for k in keepScans}

        return outmat

    def loadPtychoReconstruction(self, filepath):
        """Load all parts of the ptycho reconstruction: the object, 
        probe, and parameters p"""

        h5 = h5py.File(filepath, 'r')
        probe = h5['/reconstruction/probes'][:, :]
        object = h5['/reconstruction/object'][:, :]
        pTmp = h5['/reconstruction/p']
        p = {}  # Load p-data into a dict
        for field in pTmp.keys():
            if type(pTmp[field]) == h5py._hl.dataset.Dataset:
                p[field] = pTmp[field][()]
        return probe, object, p

    def loadSlice(self, scanNum):
        "Load a single projection"

        filepath = self.projFilepaths[scanNum]
        h5 = h5py.File(filepath, 'r')
        sliceObject = h5['/reconstruction/object'][:, :].astype(np.complex64)
        return sliceObject

    def parLoadStackObject(self, scanNums, numProcesses=int(mp.cpu_count()*0.8)):
        "Open a process pool to load the full stack of projections"

        pool = mp.Pool(numProcesses)
        try:
            print("Loading projections into list...")
            t0 = lam.utils.timerStart()
            stack = pool.starmap(self.loadSlice, zip(scanNums))
            lam.utils.timerEnd(t0, "Loading projections into list...Completed", True)
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            print(traceback.format_exc())
        finally:
            pool.close()  # prevent more processes from being submitted
            pool.join()  # wait for processes to end
            pool.terminate()
            del pool

        return stack

    def getScanPositions(self, loadFrom='reconstructions'):
        """Load the scan positions for each projection file. The 
        variable `loadFrom` indicates where the scan positions were 
        loaded from."""

        if loadFrom == 'omny':
            # These positions are rotated 90 degrees compared to the 
            # positions in the 'reconstruction' format. These will NOT
            # work for creating weights due to the varying file formats.
            scanPositions = []
            for i in range(len(self.scansTomo)):
                positionPath = self.OmnyPosFile + "/scan_0" + str(self.scansTomo[i]) + ".dat"
                tmp = pd.read_csv(positionPath, delim_whitespace=True, header=[1])
                # Do some conversions to match the 'reconstructions'
                scanPositions.append(np.array([tmp['Average_x_st_fzp'],
                                               tmp['Average_y_st_fzp']]).transpose())
        elif loadFrom == 'reconstructions':
            # This type of scan position should be compatible with the 
            # function for getting weights, but should be checked if 
            # you get weights that way
            scanPositions = []
            for i in self.projFilepaths.keys():
                if type(self) is lam.read.fileReader:
                    f = h5py.File(self.projFilepaths[i], 'r')
                    scanPositions.append(f['/reconstruction/p/positions_real'][:])
                    f.close()
                # elif type(self) is lam.read.fileReader_mat:
                elif issubclass(type(self), lam.read.fileReader_mat):
                    try:
                        f = sio.loadmat(self.projFilepaths[i], variable_names='outputs')
                        scanPositions.append(f['outputs']['probe_positions'][0, 0])
                    except Exception:
                        scanPositions.append([])

                    if np.round(i % 10) == i % 10:
                        print("Loaded scan positions for projection " + str(i))

        return scanPositions, loadFrom


# there are different file reader classes for different file structures
class fileReader_mat(fileReader):
    def __init__(self, config):

        self.scansTomo = np.arange(int(config['File Info']['scanStart']),
                                   int(config['File Info']['scanEnd']))
        self.basePath = config['File Info']['basePath']
        self.analysisFolder = config['File Info']['analysisFolder']
        self.filePrefix = config['File Info']['filePrefix']
        self.fileSuffix = config['File Info']['fileSuffix']
        self.roi = config['File Info']['roi']
        self.method1 = config['File Info']['method1']
        self.method2 = config['File Info']['method2']
        self.Niter = config['File Info']['Niter']
        self.fileExtension = config['File Info']['fileExtension']

        self.analysisPath = self.basePath + self.analysisFolder
        self.OmnyAngleFile = self.basePath + config['File Info']['scanNumbersPath']
        self.OmnyPosFile = self.basePath + config['File Info']['positionsPath']

        if config['File Info']['removeScans'] != '':
            self.removeScans = [int(i) for i in config['File Info']['removeScans'].split(',')]
        else:
            self.removeScans = []

        self.findProjectionFilepaths()

    def findProjectionFilepaths(self, scanNums=[]):
        """Find filepath for the ptychography reconstruction for the 
        first scan"""

        if scanNums == []:
            scanNums = self.scansTomo
        removeIdx = np.array([], dtype=int)

        self.projFilepaths = {}
        for scanNum in scanNums:
            fileFound = False
            # Find the file to load
            for string in os.listdir(self.analysisPath):
                try:
                    scanString = re.findall('S(\w*)', string)[0]
                    if int(scanString) == scanNum:
                        filepath = os.path.join(self.analysisPath + 'S' + scanString, 
                                                self.roi, 
                                                self.method1,
                                                self.method2, 
                                                self.Niter + '.' + self.fileExtension)
                        filepath = glob.glob(filepath)
                        if filepath != []:
                            fileFound = True
                            self.projFilepaths[scanNum] = filepath[0]
                except:
                    pass
            if fileFound == False:
                print("No projection data found for scan", scanNum)
                removeIdx = np.append(removeIdx, np.where(self.scansTomo == scanNum))
            else:
                if fileFound == False:
                    print("No projection data found for scan", scanNum)
                else: 
                    print("Found scan", str(scanNum) + ":", filepath)
        self.scansTomo = np.delete(self.scansTomo, removeIdx)

    def loadSlice(self, scanNum):
        "Load a single projection"

        filepath = self.projFilepaths[scanNum]
        try:
            f = h5py.File(filepath, 'r')
            sliceObject = f['object'][:]
            sliceObject = (sliceObject['real'] +
                           1j*sliceObject['imag']).astype(np.complex64)
            f.close()
        except:
            if 'f' in locals():
                f.close()
            try:
                f = sio.loadmat(filepath, variable_names=['object'])
                sliceObject = f['object'].astype(np.complex64)
            except:
                sliceObject = []
        finally:
            pass

        return sliceObject

    def loadPtychoReconstruction(self, filepath):

        try:
            fileContents = h5py.File(filepath, 'r')
            probe = fileContents['probe'][:]
            object = fileContents['object'][:]
            p = {}  # Load parameter-data into a dict
            # parameters p is made to match h5 formatting
            p['asize'] = np.array(probe.shape[0:2])[:, np.newaxis]
            p['dx_spec'] = fileContents['p']['dx_spec'][:][0][0]
            p['lambda'] = fileContents['p']['lambda'][:][0][0]
            fileContents.close()
        except:
            if 'fileContents' in locals():
                fileContents.close()
            fileContents = sio.loadmat(filepath)
            probe = fileContents['probe']
            probeSum = np.zeros(probe[0][0].shape)
            for i in range(len(probe)):
                probeSum = probeSum + probe[i][0]
            probe = probeSum
            object = fileContents['object']
            p = {}  # Load parameter-data into a dict
            p['dx_spec'] = fileContents['p']['dx_spec'][0, 0][0]
            p['lambda'] = fileContents['p']['lambda'][0, 0][0][0]

        return probe, object, p


class fileReader_mat_new(fileReader_mat):

    def __init__(self, config):
        self.scansTomo = np.arange(int(config['File Info']['scanStart']),
                                   int(config['File Info']['scanEnd']))
        self.basePath = config['File Info']['basePath']
        self.analysisFolder = config['File Info']['analysisFolder']
        self.filePattern = config['File Info']['filePattern']
        self.analysisPath = os.path.join(self.basePath, self.analysisFolder)
        self.OmnyAngleFile = os.path.join(self.basePath, config['File Info']['scanNumbersPath'])
        self.OmnyPosFile = os.path.join(self.basePath, config['File Info']['positionsPath'])

        if config['File Info']['removeScans'] != '':
            self.removeScans = [int(i) for i in config['File Info']['removeScans'].split(',')]
        else:
            self.removeScans = []

        self.findProjectionFilepaths()

    def findProjectionFilepaths(self, scanNums=[]):
        """Find filepath for the ptychography reconstruction for the 
        first scan"""

        if scanNums == []:
            scanNums = self.scansTomo
        removeIdx = np.array([], dtype=int)

        self.projFilepaths = {}
        for scanNum in scanNums:
            fileFound = False
            # Find the file to load
            for string in os.listdir(self.analysisPath):
                try:
                    scanString = re.findall('S(\w*)', string)[0]
                    if int(scanString) == scanNum:
                        filepath = os.path.join(self.analysisPath + 'S' + scanString, 
                                                self.filePattern)
                        filepath = glob.glob(filepath)
                        if filepath != []:
                            fileFound = True
                            self.projFilepaths[scanNum] = filepath[0]
                except:
                    pass
            if fileFound == False:
                print("No projection data found for scan", scanNum)
                removeIdx = np.append(removeIdx, np.where(self.scansTomo == scanNum))
        self.scansTomo = np.delete(self.scansTomo, removeIdx)
