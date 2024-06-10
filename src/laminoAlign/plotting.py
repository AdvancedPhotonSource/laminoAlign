import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import multiprocessing as mp
from itertools import repeat
import os
import subprocess

default_crf = 20


def animatePlot(frames, title="", xlabel="", ylabel="", saveGIF=False, filename="", FPS=10,
                DPI=100, frameSkip=1):
    """Method for making movies from 2-dimensional arrays.

    See `laminoAlign.projectionMatching.saveMovies` for examples."""

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x = np.arange(0, frames.shape[1])
    y = frames
    ln, = ax.plot(x, y[0])
    plt.ylim(np.min(frames), np.max(frames))

    def init():
        plt.ylim(np.min(frames), np.max(frames))
        return ln,

    def update(num):
        ln.set_data(x, y[num])
        ax.set_title(title + " " + str(frameSkip*num))
        return ln

    ani = FuncAnimation(fig, func=update, frames=y.shape[0])

    if saveGIF:
        ani.save(filename+".gif", dpi=DPI, writer=PillowWriter(fps=FPS))

    return ani


def animateStack(frames, title="", xlabel="", ylabel="", clim=[], plotType="plot", filename="",
                 FPS=10, DPI=100, frameSkip=1, imgFormat="png", colormap="bone", crf=default_crf,
                 overwriteImages=True):
    """Wrapper method for saving movies of a a stack of images (a 3D 
    array).

    See See `laminoAlign.projectionMatching.saveMovies` for examples."""

    plt.rcParams['image.cmap'] = colormap

    if plotType == "avi" or plotType == "mp4":
        saveStackMovie(frames, title, xlabel, ylabel, clim, filename, FPS,
                       DPI, imgFormat, plotType, crf, overwriteImages)
    elif plotType == "gif":
        saveGIF(frames, title, xlabel, ylabel, clim, filename, FPS,
                DPI, frameSkip, saveGIF=True)
    elif plotType == "plot":
        return saveGIF(frames, title, xlabel, ylabel, clim, filename, FPS,
                       DPI, frameSkip, saveGIF=False)


def saveStackMovie(frames, title="", xlabel="", ylabel="", clim=[], filename="", FPS=10, DPI=100,
                   imgFormat="png", movieFormat="avi", crf=default_crf, overwriteImages=True):

    imageFolder = filename + " images"
    if not os.path.isdir(imageFolder):
        os.makedirs(imageFolder)

    if overwriteImages or (len(os.listdir(imageFolder)) == 0):

        # Use 50 % of the available CPUs to make the movies
        Np = int(mp.cpu_count()*0.5)
        pool = mp.Pool(Np)

        numPad = len(str(frames.shape[0]))
        numString = [("{:0" + str(numPad)+"d}").format(i) for i in range(frames.shape[0])]

        inputs = zip(frames, numString, repeat(imageFolder), repeat(title),
                     repeat(clim), repeat(DPI), repeat(imgFormat))
        pool.starmap(stackMovieProcess, inputs)

        # End the pool
        pool.close()
        pool.join()
        pool.terminate()

    # Save PNGs to a movie
    moviePath = filename + "." + movieFormat
    filePattern = os.path.join(imageFolder, '*.' + imgFormat)

    command = ("ffmpeg -y -pattern_type glob -framerate " + str(FPS) + " -i '" + filePattern + "' "
               "-c:v libx264 -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white\" "
               "-crf " + str(crf) + " -preset veryslow '" + moviePath + "' -hide_banner -loglevel error")
    subprocess.run(command, shell=True)


def stackMovieProcess(frame, numString, folder, title, clim, DPI, imgFormat):
    try:
        fig, ax = plt.subplots()
        plt.imshow(frame, aspect='equal')
        plt.title(title + " " + numString)
        if clim != []:
            plt.clim(clim)
        plt.savefig(os.path.join(folder, "image " + numString + "." + imgFormat), format=imgFormat, dpi=DPI)
    except Exception as ex:
        print("Exception occurred while saving image...")
        print("Exception type:", type(ex).__name__)
        print("Exception arguments:", ex.args)
    finally:
        plt.close()
        return


def saveGIF(frames, title="", xlabel="", ylabel="", clim=[], filename="", FPS=10, DPI=100,
            frameSkip=1, saveGIF=True):

    fig, ax = plt.subplots()
    ln = plt.imshow(frames[0, :, :], aspect='equal')

    if clim != []:
        ln.set_clim(clim)

    def update(num):
        ln.set_data(frames[num])
        ax.set_title(title + " " + str(frameSkip*num))
        return ln,

    ani = FuncAnimation(fig, func=update, frames=frames.shape[0])
    if saveGIF:
        ani.save(filename+".gif", dpi=DPI, writer=PillowWriter(fps=FPS))

    return ani


def plotImage(img, widths=[], xCenter=None, yCenter=None, ax=""):

    if ax != "":
        plt.axes(ax)

    plt.imshow(img)
    if not (type(widths) is np.ndarray or type(widths) is list):
        widths = [widths, widths]
    if len(widths) == 2:
        if xCenter is None:
            xCenter = img.shape[1]/2
        if yCenter is None:
            yCenter = img.shape[0]/2
        xlim = xCenter + np.array([-widths[0]/2, widths[0]/2])
        ylim = yCenter + np.array([-widths[1]/2, widths[1]/2])
        plt.xlim(xlim)
        plt.ylim(ylim)
    return plt
