import numpy as np
import scipy.linalg as linalg

def M_RFF_Generator(X, Omega, a=1.0):
    """
        Compute feature matrices for RFF approximation of the Koopman generator
        for an SDE. The modified version is based on real-valued features, using
        sin and cos functions replacing the exponentials. 

        Parameters:
        --------
        X (d, m):           data array, m points in d-dimensional space.
        Omega (d, p):       feature array, p point in d-dimensional frequency space
        a:                  diffusion tensor at all data sites. a can be
                            float or (d, d, m).

        Return:
        --------
        M (m, p):       evaluation of Fourier features all data point
        ML (m, p) or (p, p):
                        application of generator appplied to all Fourier features at all
                        data points. If reversible is True, ML is the contraction of the
                        gradients of all Fourier features with the diffusion field, which
                        is a pxp-matrix.
    """
    # Compute Feature feature matrix:
    arg = np.dot(X.T, Omega)
    M = np.hstack((np.cos(arg),np.sin(arg))) 

    if isinstance(a, float) or isinstance(a, int):
        DV = np.concatenate((np.einsum("ij, gj -> ijg", Omega, -np.sin(arg)),np.einsum("ij, gj -> ijg", Omega, np.cos(arg))),axis=1)
        ML_aux = (0.5) * DV * a
        ML = np.tensordot(ML_aux, DV, axes=[(0, 2), (0, 2)])
    else:
        DV = np.concatenate((np.einsum("ij, gj -> ijg", Omega, -np.sin(arg)),np.einsum("ij, gj -> ijg", Omega, np.cos(arg))),axis=1)
        ML_aux = (0.5) * np.einsum("irs, ijs -> jrs", DV, a)
        ML = np.tensordot(ML_aux, DV, axes=[(0, 2), (0, 2)])

    return M, ML

def eigen_decomposition(G0,G1,tol,r_max):
    """
    Compute eigenvalues and eigenvectors of the generator matrix.

    tol, float:
        cut-off for small singular values of mass matrix.


    Returns:
    --------
    kappa, ndarray(nev,)
        array of eigenvalues, sorted in ascending order.
    V, ndarray(nf, nev)
        array of corresponding eigenvectors.

    """
    # Compute transformed generator matrix and transformation matrix:
    U, L = compute_UL(G0, G1, tol, r_max)
    # Diagonalize L:
    kappa, V = linalg.eigh(L)   
    # Transform eigenvectors back to original basis:
    V = np.dot(U, V)

    return kappa, V, U, L

def compute_UL(G0,G1,tol,r_max):
    """
    Compute transformation U to orthonormal basis and generator matrix L.

    tol, float:
        cut-off for small singular values of mass matrix.
    
    Returns:
    --------
    U, ndarray (nf, r):
        transformation to orthonormal basis (possible of smaller size)
    L, ndarray (r, r):
        generator matrix for transformed basis

    """
    # Eigenvalue decomposition:
    s, U = linalg.eigh(G0)

    # Sort in decreasing order
    s = s[::-1]
    U = U[:, ::-1]

    # Elminate small singular values:
    ind = np.where(s / s[0] >= tol)[0]

    s = s[ind]
    U = U[:, ind]

    r = np.minimum(U.shape[1], r_max) # ROM size
    U = U[:,:r]
    s = s[:r]

    # Transform basis:
    U = U * (s ** (-0.5))

    # Compute projected generator matrices:
    L = np.dot(U.T, np.dot(G1, U))
    
    return U, L

