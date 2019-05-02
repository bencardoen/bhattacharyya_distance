from numba import jit
import numpy as np

@jit(parallel=True)
def bhattacharyya(x,y):
    covx = np.cov(x, rowvar=False)
    covy = np.cov(y, rowvar=False)
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    sigm = (covx + covy)/2
    ds = np.linalg.det(sigm)
    dcx = np.linalg.det(covx)
    dcy = np.linalg.det(covy)
    firstp = 1/8.0 * np.matmul(np.transpose((mx - my)), np.linalg.inv(sigm))
    first = np.matmul(firstp, (mx-my))
    second = 0.5 * np.log(ds / np.sqrt(dcx*dcy))
    db = first + second
    return db
