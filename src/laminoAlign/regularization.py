import numpy as np
import cupy as cp
import laminoAlign as lam
import traceback


def div(P):

    xp = cp.get_array_module(P)

    Px = P[0, :, :, :]
    Py = P[1, :, :, :]
    Pz = P[2, :, :, :]

    idx = xp.array([0] + [i for i in range(P.shape[2]-1)])
    fx = Px - Px[:, idx, :]
    idx = xp.array([0] + [i for i in range(P.shape[3]-1)])
    fy = Py - Py[:, :, idx]
    idx = xp.array([0] + [i for i in range(P.shape[1]-1)])
    fz = Pz - Pz[idx, :, :]

    fd = fx + fy + fz
    return fd


def grad(M):

    xp = cp.get_array_module(M)

    idx = np.array([i+1 for i in range(M.shape[1]-1)] + [M.shape[1]-1])
    fx = M[:, idx, :] - M
    idx = np.array([i+1 for i in range(M.shape[2]-1)] + [M.shape[2]-1])
    fy = M[:, :, idx] - M
    idx = np.array([i+1 for i in range(M.shape[0]-1)] + [M.shape[0]-1])
    fz = M[idx, :, :] - M

    f = np.stack([fz, fx, fy])
    return f


def chambolleLocalTV3D(x, alpha, Niter):

    try:
        xp = cp.get_array_module(x)

        (L, M, N) = x.shape
        x0 = x
        xi = xp.zeros((3, L, M, N), dtype=np.float32)
        tau = 1/4

        for i in range(Niter):
            # Chambolle step
            t0 = lam.utils.timerStart()
            gdv = grad(div(xi) - x/alpha)
            lam.utils.timerEnd(t0, "Chambolle Step")

            # Anisotropic
            t0 = lam.utils.timerStart()
            d = xp.abs(gdv.sum(axis=0))
            xi = (xi + tau*gdv) / (1+tau*d)
            lam.utils.timerEnd(t0, "Anisotropic")

            # Reconstruct
            t0 = lam.utils.timerStart()
            x = x - alpha*div(xi)
            print("iteration", i)
            lam.utils.timerEnd(t0, "Reconstruct")

        # prevent pushing values to zero by the TV regularization
        x = xp.sum(x0*x)/xp.sum(x**2) * x
        return x

    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        traceback.print_exc()
    finally:
        lam.utils.freeAllBlocks()
