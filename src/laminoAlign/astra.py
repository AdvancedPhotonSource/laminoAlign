import numpy as np
import astra
import laminoAlign as lam


def initializeAstra(sinogram, angles, Npix, laminoAngle, tiltAngle=0, skewAngle=0,
                    CoROffset=[0, 0], pixelScale=[1, 1]):

    cfg = {}
    cfg['iVolX'] = Npix[0]
    cfg['iVolY'] = Npix[1]
    cfg['iVolZ'] = Npix[2]
    cfg['iProjAngles'] = len(angles)
    cfg['iProjU'] = sinogram.shape[2]
    cfg['iProjV'] = sinogram.shape[1]
    cfg['iRaysPerDet'] = 1
    cfg['iRaysPerDetDim'] = 1
    cfg['iRaysPerVoxelDim'] = 1
    sourceDistance = 1

    # Get projection geometry
    angles = (angles + 90)*np.pi/180
    laminoAngle = (np.pi/2 - laminoAngle*np.pi/180)*np.ones((len(angles), 1))
    tiltAngle = np.pi/180*tiltAngle*np.ones((len(angles), 1))
    skewAngle = np.pi/180*skewAngle*np.ones((len(angles), 1))
    pixelScale = pixelScale*np.ones((len(angles), 2))
    rotationCenter = np.array([CoROffset[1], CoROffset[0]], dtype=np.float32)

    # We generate the same geometry as the circular one above.
    vectors = np.zeros((len(angles), 12))

    laminoAngle = laminoAngle.transpose()

    # https://www.astra-toolbox.com/docs/geom3d.html
    # Vectors: (rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ)

    # ray direction
    vectors[:, 0] = np.sin(angles)*np.cos(laminoAngle)
    vectors[:, 1] = -np.cos(angles)*np.cos(laminoAngle)
    vectors[:, 2] = np.sin(laminoAngle)
    vectors[:, [0, 1, 2]] = vectors[:, [0, 1, 2]]*sourceDistance

    # center of detector
    vectors[:, [3, 4, 5]] = 0

    # vector from detector pixel (0,0) to (0,1)
    vectors[:, 6] = np.cos(angles)/pixelScale[:, 0]
    vectors[:, 7] = np.sin(angles)/pixelScale[:, 0]
    vectors[:, 8] = 0/pixelScale[:, 0]

    # vector from detector pixel (0,0) to (1,0)
    vectors[:, 9] = -np.sin(laminoAngle)*np.sin(angles)/pixelScale[:, 1]
    vectors[:, 10] = np.sin(laminoAngle)*np.cos(angles)/pixelScale[:, 1]
    vectors[:, 11] = np.cos(laminoAngle)/pixelScale[:, 1]

    # Center offset alignment
    vectors[:, 3:6] = vectors[:, 3:6] - (vectors[:, 9:12]*(rotationCenter[0])
                                         + vectors[:, 6:9]*(rotationCenter[1]))

    # Apply Rodrigues' rotation formula to rotate and skew detector
    # Apply tilt angle: rotate detector in plane perpendicular to the beam axis
    for i in range(len(angles)):
        vectors[i, 6:9] = (
            vectors[i, 6:9]*np.cos(tiltAngle[i])
            + np.cross(vectors[i, 0:3], vectors[i, 6:9])
            * np.sin(tiltAngle[i])
            + vectors[i, 0:3]*np.dot(vectors[i, 0:3], vectors[i, 6:9])
            * (1 - np.cos(tiltAngle[i]))
        )
        vectors[i, 9:12] = (
            vectors[i, 9:12]*np.cos(tiltAngle[i])
            + np.cross(vectors[i, 0:3], vectors[i, 9:12])
            * np.sin(tiltAngle[i])
            + vectors[i, 0:3]*np.dot(vectors[i, 0:3], vectors[i, 9:12])
            * (1 - np.cos(tiltAngle[i]))
        )

    # Apply skew angle: rotate only one axis of the detector
    for i in range(len(angles)):
        vectors[i, 9:12] = (
            vectors[i, 9:12]*np.cos(skewAngle[i]/2)
            + np.cross(vectors[i, 0:3], vectors[i, 9:12])
            * np.sin(skewAngle[i]/2)
            + vectors[i, 0:3]*np.dot(vectors[i, 0:3], vectors[i, 9:12])
            * (1 - np.cos(skewAngle[i]/2))
        )

    return cfg, vectors


def getGeometries(cfg, vectors):

    geometries = {}
    geometries['vol_geom'] = astra.create_vol_geom(
        cfg['iVolX'],
        cfg['iVolY'],
        cfg['iVolZ']
    )
    geometries['proj_geom'] = astra.create_proj_geom(
        'parallel3d_vec',
        cfg['iProjV'],  # detector rows
        cfg['iProjU'],  # detector columns
        vectors
    )

    return geometries


def astraReconstruct(sinogram, cfg, vectors, geometries={}, astraConfig={}, timerOn=False):
    """Get the 3D reconstruction"""

    t0 = lam.utils.timerStart()
    if astraConfig == {}:
        geometries = getGeometries(cfg, vectors)
        astraConfig = astra.astra_dict('BP3D_CUDA')
        astraConfig['ReconstructionDataId'] = astra.data3d.create('-vol', geometries['vol_geom'])
        astraConfig['ProjectionDataId'] = astra.data3d.create('-sino', geometries['proj_geom'],
                                                              sinogram.transpose([1, 0, 2]))
    else:
        geometries = getGeometries(cfg, vectors)
        astra.data3d.change_geometry(astraConfig['ReconstructionDataId'], geometries['vol_geom'])
        astra.data3d.change_geometry(astraConfig['ProjectionDataId'], geometries['proj_geom'])
        astra.data3d.store(astraConfig['ProjectionDataId'], sinogram.transpose([1, 0, 2]))
    lam.utils.timerEnd(t0, "Astra: Create Geometry and Objects", timerOn)

    t0 = lam.utils.timerStart()
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(astraConfig)
    lam.utils.timerEnd(t0, "Astra: Create Algorithm", timerOn)

    t0 = lam.utils.timerStart()
    # Run the reconstruction algorithm
    astra.algorithm.run(alg_id)
    astra.algorithm.clear()
    lam.utils.timerEnd(t0, "Astra: Run Algorithm", timerOn)

    t0 = lam.utils.timerStart()
    # Retrieve the reconstruction
    rec = astra.data3d.get(astraConfig['ReconstructionDataId'])
    lam.utils.timerEnd(t0, "Astra: Retrieve Reconstruction", timerOn)

    return rec, astraConfig, geometries
