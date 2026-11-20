import numpy as np
from scipy.linalg import svd, eig, eigvals
from scipy.special import iv
from sklearn.model_selection import train_test_split


def M_RFF_Generator(X, Omega, grad_V, beta, gamma, mass):
    """
    Compute feature matrices for RFF approximation of the Koopman generator
    for an SDE.

    Parameters
    --------
    X (d, m):           data array, m points in d-dimensional space.
    Omega (d, p):       feature array, p point in d-dimensional frequency space
    beta () :           Inverse temperature
    grad_V (d, m):      Potential gradient with respect to the position states 

    Returns
    --------
    M0 (m, p):       evaluation of Fourier features all data point
    A (p, p):
                    application of generator appplied to all Fourier features at all
                    data points. ML is the contraction of the gradients of all Fourier
                    features with the diffusion field, which is a pxp-matrix.
    """
    d = int(X.shape[0]/2)

    arg = np.dot(X.T, Omega)

    M0 = np.hstack((np.cos(arg),np.sin(arg))) 

    DV = np.concatenate((np.einsum("ij, gj -> ijg", Omega, -np.sin(arg)),np.einsum("ij, gj -> ijg", Omega, np.cos(arg))),axis=1)

    if np.isscalar(gamma):

        A_ou = -(gamma/beta) * np.einsum("ghj, gij -> hi", DV[d:], DV[d:]*mass[:, None, None])

    else:

        A0 = np.einsum("fij, fgj -> gij", DV, gamma)

        A_ou = -1/beta * np.einsum("ghj, gij -> hi", A0, A0*mass[:, None, None]) 


    A_h_2 = np.einsum("gj, ghj -> hj", -grad_V, DV[d:])

    A_h_1 = np.einsum("gj, gij -> ij", X[d:]/mass[:, None], DV[:d])

    A_h_aux =  A_h_1 + A_h_2

    A_H = np.einsum("ig, ji -> jg", M0, A_h_aux)

    return M0, A_ou, A_H

def M_RFF_Generator_coarse_grained(cg_data, P, Omega, grad_V, cg_map_grad, beta, gamma, weights, mass):
    """
    Compute feature matrices for RFF approximation of the Koopman generator
    for an SDE.

    Parameters
    --------
    cg_data (d_zeta, m) coarse-grained data array, m points in d_zeta-dimensional space
    P (d, m):           full-space momentum, m points in d-dimensional space.
    Omega (2k, p):      feature array, p point in k-dimensional frequency space
    grad_V (d, m):      potential gradient with respect to the position states 
    cg_map_grad (2k, 2d, m) :   gradients of the coarse-graining map at all data points
                        units are [[nm^{-1}], 0,
                                    [ps^{-1} nm^{-1}], [nm^{-1} amu^{-1}]]
    beta () :           Inverse temperature
    gamma ():           Friction term
    weights (m,)        Reweighting weights of the samples

    Returns
    --------
    M0 (m, p):       evaluation of Fourier features all data point
    A (p, p):
                    application of generator appplied to all Fourier features at all
                    data points. ML is the contraction of the gradients of all Fourier
                    features with the diffusion field, which is a pxp-matrix.
    """
    d = int(P.shape[0])
    k = int(cg_data.shape[0]/2)

    # Compute Feature feature matrix:
    arg = np.dot(cg_data.T, Omega)

    M0 = np.hstack((np.cos(arg),np.sin(arg))) 
    DV = np.concatenate((np.einsum("ij, gj -> ijg", Omega, -np.sin(arg)),np.einsum("ij, gj -> ijg", Omega, np.cos(arg))),axis=1)

    # Computing p-gradients of random features... (d x p x m):
    A0 = np.einsum("fij, fgj -> gij", DV[k:], cg_map_grad[k:,d:])

    # Computing diffusion part of the generator...
    if np.isscalar(gamma):
        A_ou = -gamma/(beta) * np.einsum("ghj, gij -> hi", A0*weights, A0*mass[:, None, None])
    else:
        A0_aux = np.einsum("fij, fgj -> gij", DV, gamma)
        A_ou = -1/beta * np.einsum("ghj, gij -> hi", A0_aux*weights, A0_aux*mass[:, None, None]) 


    # Computing force term in Hamiltonian generator... (p x m)
    A_h_2 = np.einsum("gj, ghj -> hj", -grad_V, A0)

    # Computing q-gradients of random features... (d x p x m):
    A1 = np.einsum("gij, ghj -> hij", DV, cg_map_grad[:, :d])

    # Computing velocity term in Hamiltonian generator... (p x m)
    A_h_1 = np.einsum("gj, gij -> ij", P/mass[:, None], A1)

    # Add them up to get the total Hamiltonian part of the generator:
    A_H_aux =  A_h_1 + A_h_2

    # Multiply out with random features to compute matrix rep for Hamiltonian part (p x p):
    A_H = np.einsum("ig, ji -> jg", M0, A_H_aux*weights)

    return  M0* np.sqrt(weights.T), A_ou, A_H

def spectral_analysis_rff_generator(X, Omega, nev, grad_V, beta, gamma, tol, batch_size=None, mass=None):
    """
    Compute spectral decomposition of the Koopman generator
        based on random Fourier features.

    Parameters
    -----------
    X, array (d, m) or list of arrays(d, m_i)
                    full data, m points in d-dimensional space
    Omega, (d, p):  random Fourier features in d-dimensional space.
    nev, int:       number of eigenvalues to be retained
    a, float or array(d, d, m) or list of arrays(d, d, m_i):
                    diffusion tensor at all data points.
    tol, float:     svd cutoff
    batch_size, int or None:
                    process data in chunks to avoid excessive memory usage.

    Returns
    -------
    d, (nev,)       eigenvalues of the Koopman generator
    W, (p, nev)     corresponding eigenvectors
    M, (m, p)   random Fourier feature matrix at all data points

    """
    # Concatenate data if given as lists:
    if isinstance(X, list):
        X = np.hstack(X)
    if mass is None:
        mass = np.ones((X.shape[0]//2,))
    if batch_size is None:
        batch_size = X.shape[1]

    nbatch = int(np.ceil(X.shape[1] / batch_size))

    # Generate Fourier matrices:
    M = np.zeros((0, 2*Omega.shape[1]))
    ML1 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))
    ML2 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))

    for ii in range(nbatch):
        indxs = np.s_[ii*batch_size : (ii+1)*batch_size]
        if np.isscalar(gamma):
            Mi, A_ou, A_H = M_RFF_Generator(X[:, indxs], Omega, grad_V[:, indxs],
                                    beta, gamma, mass)
        else:
            Mi, A_ou, A_H = M_RFF_Generator(X[:, indxs], Omega, grad_V[:, indxs],
                                    beta, gamma[:,:, indxs], mass)
        # Update generator matrices:
        ML1 += A_ou
        ML2 += A_H
        M = np.vstack((M, Mi))

    ML2 = (ML2 - ML2.T)/2
    ML = ML1 + ML2
    # Compute whitening transformation:
    L = whitening_transform(M.T, tol, rmin=nev)
    # Compute reduced matrix
    R = L.T @ ML @ L
    # Solve eigenvalue problem:
    di, Wi = eig(R)
    # Filter and sort eigenvalues and eigenvectors:
    di, Wi = filter_ev(di, Wi)
    # Compute transformation to eigenvector space:
    W = L @ Wi[:, -nev:]

    return di, W, L, M
    
def spectral_analysis_rff_generator_coarse_grained(cg_data, grad_V, cg_map_grad, P, Omega, nev, beta, gamma, 
                                                    tol, batch_size=None, weights=None, mass=None):
    """
    Compute spectral decomposition of the Koopman generator
        based on random Fourier features.

    Parameters
    -----------
    cg_data, array (d_zeta, m):
                    Coarse-grained data, m points in d_zeta-dimensional space
    grad_V, array(d, m) or list of arrays(d, m_i):
                    force at all data points.
    P, array (d, m) or list of arrays(d, m_i)
                    full-space momentum, m points in d-dimensional space
    Omega, (d, p):  random Fourier features in d-dimensional space.
    nev, int:       number of eigenvalues to be retained
    tol, float:     svd cutoff
    batch_size, int or None:
                    process data in chunks to avoid excessive memory usage.

    Returns
    -------
    d, (nev,)       eigenvalues of the Koopman generator
    W, (p, nev)     corresponding eigenvectors
    M, (m, p)   random Fourier feature matrix at all data points

    """
    # Concatenate data if given as lists:
    if isinstance(P, list):
        P = np.hstack(P)
    if weights is None:
        weights = np.ones((1,P.shape[1]))
    # Check if batch_size is given:
    if batch_size is None:
        batch_size = P.shape[1]
    if mass is None:
        mass = np.ones((P.shape[0],))
    # Determine number of batches:
    nbatch = int(np.ceil(P.shape[1] / batch_size))
    
    # Generate Fourier matrices:
    M = np.zeros((0, 2*Omega.shape[1]))
    ML1 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))
    ML2 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))

    for ii in range(nbatch):
        indxs = np.s_[ii*batch_size : (ii+1)*batch_size]
        if np.isscalar(gamma):
            Mi, A_ou, A_H = M_RFF_Generator_coarse_grained(cg_data[:, indxs], P[:, indxs],
                                Omega, grad_V[:, indxs], cg_map_grad[:, :,indxs], 
                                beta, gamma, weights[:,indxs], mass)
        else:
            Mi, A_ou, A_H = M_RFF_Generator_coarse_grained(cg_data[:, indxs], P[:, indxs],
                                Omega, grad_V[:, indxs], cg_map_grad[:, :, indxs], 
                                beta, gamma[:,:, indxs], weights[:, indxs], mass)
            
        # Update generator matrices:
        ML1 += A_ou
        ML2 += A_H
        M = np.vstack((M, Mi))

    ML2 = (ML2 - ML2.T)/2
    ML = (ML1 + ML2)
    # Compute whitening transformation:
    L = whitening_transform(M.T, tol, rmin=nev, return_svd=False, complex_val=False)
    # Compute reduced matrix
    R = L.T @ ML @ L    
    # Solve eigenvalue problem:
    di, Wi = eig(R)
    # Filter and sort eigenvalues and eigenvectors:
    di, Wi = filter_ev(di, Wi)
    # Compute transformation to eigenvector space:
    W = L @ Wi[:, -nev:]
    
    return di, W, L, M

def cv_generator_rff(P, Omega, beta, gamma, grad_V, rtrain, ntest, nev, tol, batch_size=None, 
                     cg_data=None, cg_map_grad=None, weights=None, mass=None):   
    """
    Score RFF model for the Koopman generator using repeated random shuffling of the data.

    Parameters
    -----------
    X, array (d, m) or list of arrays(d, m_i)
                    full data, m points in d-dimensional space
    Omega, (d, p):  random Fourier features in d-dimensional space
    beta, float or array(d, d, m) or list of arrays(d, d, m_i):
                    diffusion tensor at all data points.
    rtrain, float:  ratio of training vs test data
    ntest, int:     number of random shuffles
    nev, int:       number of eigenvalues for VAMP score
    tol, float:     svd cutoff

    Returns
    -------

    """
    # Concatenate data if given as lists:
    if isinstance(P, list):
        P = np.hstack(P)
    if weights is None:
        weights = np.ones((1,P.shape[1]))
    if batch_size is None:
        batch_size = P.shape[1]
    if mass is None:
        mass = np.ones((P.shape[0],))
    # Prepare output:
    d = np.zeros((ntest, nev), dtype=complex)
    dtest = np.zeros((ntest, nev), dtype=complex)
    # Compute test score for each random shuffle of the data:
    for ii in range(ntest):
        # Random shuffling of the data:
        if cg_data is not None:
            Ptrain, Ptest, cg_datatrain, cg_datatest, cg_map_gradtrain, cg_map_gradtest, \
            grad_Vtrain, grad_Vtest, weights_train, weights_test = train_test_split(P.T, cg_data.T,
                                        np.transpose(cg_map_grad, (2,0,1)), grad_V.T, 
                                        weights.T, train_size=rtrain)

            Ptrain, Ptest, cg_datatrain, cg_datatest, cg_map_gradtrain, cg_map_gradtest, grad_Vtrain, grad_Vtest,\
            weights_train, weights_test = Ptrain.T, Ptest.T, cg_datatrain.T, cg_datatest.T,\
            cg_map_gradtrain.transpose(1,2,0), np.transpose(cg_map_gradtest, (1,2,0)), \
            grad_Vtrain.T, grad_Vtest.T, weights_train.T, weights_test.T

        else:
            Ptrain, Ptest, grad_Vtrain, grad_Vtest, weights_train, weights_test = \
                train_test_split(P.T, grad_V.T, weights.T, train_size=rtrain)
            Ptrain, Ptest, grad_Vtrain, grad_Vtest, weights_train, weights_test  = \
            Ptrain.T, Ptest.T, grad_Vtrain.T, grad_Vtest.T, weights_train.T, weights_test.T

        # Evaluate RFF basis:
        M0 = np.zeros((0, 2*Omega.shape[1]))
        ML1 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))
        ML2 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))

        # Determine number of batches:
        nbatch = int(np.ceil(Ptrain.shape[1] / batch_size))
        
        if cg_data is not None:
            for jj in range(nbatch):
                indxs = np.s_[jj*batch_size : (jj+1)*batch_size]
                Mi, A_ou, A_H = M_RFF_Generator_coarse_grained(cg_datatrain[:, indxs], 
                    Ptrain[:, indxs], Omega, grad_Vtrain[:, indxs],
                    cg_map_gradtrain[:, :, indxs], beta, gamma, weights_train[:, indxs], mass)

                ML1 += A_ou
                ML2 += A_H
                M0 = np.vstack((M0, Mi))

        else:
            for jj in range(nbatch):    
                indxs = np.s_[jj*batch_size : (jj+1)*batch_size]
                Mi, A_ou, A_H = M_RFF_Generator(Ptrain[:, indxs], Omega,
                                                grad_Vtrain, beta, gamma, mass)    
                # Update generator matrices:
                ML1 += A_ou
                ML2 += A_H
                M0 = np.vstack((M0, Mi))
        ML2 = (ML2 - ML2.T)/2
        ML = ML1 + ML2
        # Compute whitening transformation:
        L = whitening_transform(M0.T, tol, rmin=nev, return_svd=False, complex_val=False)
        # Compute reduced matrix
        R = L.T @ ML @ L
        # Solve eigenvalue problem:
        di, Wi = eig(R)
        # Filter and sort eigenvalues and eigenvectors:
        di, Wi = filter_ev(di, Wi)
        # Compute transformation to eigenvector space:
        Wi = L @ Wi[:, -nev:]
        # Extract eigenvalues:
        d[ii, :] = di[-nev:]

        # Score the data:
        if cg_data is not None:
            dtest[ii, :] = _score_test_data_generator(Ptest, Omega, beta, gamma, grad_Vtest, Wi, batch_size,
                                                       weights_test, mass, cg_datatest, cg_map_gradtest)
        else:
            dtest[ii, :] = _score_test_data_generator(Ptest, Omega, beta, gamma, grad_Vtest, Wi, batch_size,
                                                      weights_test, mass)

    return d, dtest

def _score_test_data_generator(P, Omega, beta, gamma, grad_V, L, batch_size, weights, mass,
                               cg_data=None, cg_map_grad=None):
    """ 
    Compute VAMP-score for pre-selected subspace of RFF basis. This function is intended for
        hyper-parameter tuning.

    Parameters
    -----------
    Ptest, (d, m):  hold-out data used for scoring
    Omega, (d, p):  random Fourier features (should be the same as for training)
    atest, float or (d, d, m)
                    diffusion field at hold-out data sites
    L (p, r):       linear transformation to r-dimensional subspace

    Returns
    -------
    dvamp, float:   test score on the data.

    """
    # Evaluate RFF basis:
    M0 = np.zeros((0, 2*Omega.shape[1]))
    ML1 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))
    ML2 = np.zeros((2*Omega.shape[1], 2*Omega.shape[1]))
    # Determine number of batches:
    nbatch = int(np.ceil(P.shape[1] / batch_size))

    if cg_data is not None:
        for jj in range(nbatch):
            indxs = np.s_[jj*batch_size : (jj+1)*batch_size]
            Mi, A_ou, A_H = M_RFF_Generator_coarse_grained(cg_data[:, indxs], P[:, indxs], 
                                        Omega, grad_V[:, indxs], cg_map_grad[:, :, indxs], 
                                        beta, gamma, weights[:, indxs], mass)
            # Update generator matrices:
            ML1 += A_ou
            ML2 += A_H
            M0 = np.vstack((M0, Mi))
    else:
        # Process data in batches:
        for jj in range(nbatch):
            indxs = np.s_[jj*batch_size : (jj+1)*batch_size]
            Mi, A_ou, A_H = M_RFF_Generator(P[:, indxs], Omega,
                                    grad_V[:, indxs], beta, gamma, mass)    
            # Update generator matrices:
            ML1 += A_ou
            ML2 += A_H
            M0 = np.vstack((M0, Mi))
    ML2 = (ML2 - ML2.T)/2
    ML = ML1 + ML2
    # Compute reduced mass matrix:
    M1 = M0 @ L
    # Orthonormalize by SVD:
    U0, s0, _ = svd(M1.T, full_matrices=False)
    L0 = L @ (U0 * (s0 ** (-1))[None, :])
    # Compute reduced matrix:
    R = L0.T @ ML @ L0
    # Compute eigenvalues:
    dr = eigvals(R)
    # Sorting in ascending order
    ind = np.argsort(np.real(dr))
    dr = dr[ind]

    return dr

def spectral_analysis_edmd(Xfull, Omega, lag, nev, train_ratio, tol=1e-6):
    """
    Perform spectral decomposition of the Koopman operator based on finite-
    dimensional basis set.

    Parameters
    -----------
    Xfull, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    phi:            definition of the dictionary. This can either be a callable
                    function that converts a d x m-dimensional input to an n x p
                    dimensional functional time series, or a list of symbolic functions
                    of d variables. In the latter case, the symbolic variables representing
                    state space also need to be passed to this function.
    lag, int:       lag time for Koopman estimation
    nev, int:       number of eigenvalues to be retained
    tol, float:     relative svd cutoff for whitening transformation
    eps_ev, float:  discard all eigenvalue greater than 1.0 + eps_ev
    train_ratio:    

    Returns
    -------
    d, (nev,)       eigenvalues of the Koopman operator
    W, (p, nev)     corresponding eigenvectors
    L, (n,rp)       whitening transformation matrix
    """
    X, Y = split_by_lag(Xfull,lag)

    X_sampled, _, Y_sampled, _ = train_test_split(X.T, Y.T, train_size=train_ratio)
    X_sampled = X_sampled.T
    Y_sampled = Y_sampled.T

    if X_sampled.shape[1] > 1e+4:
        batch_size = 4000
    else:
        batch_size = X_sampled.shape[1]
    # Determine number of batches:
    nbatch = int(np.ceil(X_sampled.shape[1] / batch_size))
    PhiX = np.zeros((0, 2*Omega.shape[1]))   
    PhiY = np.zeros((0, 2*Omega.shape[1]))   
    for i in range(nbatch):
        arg = np.dot(X_sampled[:, i*batch_size:(i+1)*batch_size].T, Omega)
        M0 = np.hstack((np.cos(arg),np.sin(arg)))
        PhiX = np.vstack((PhiX, M0))

        arg = np.dot(Y_sampled[:, i*batch_size:(i+1)*batch_size].T, Omega)
        M0 = np.hstack((np.cos(arg),np.sin(arg)))
        PhiY = np.vstack((PhiY, M0))
    PhiX = PhiX.T
    PhiY = PhiY.T
    
    # Compute whitening transformation:
    L, usv = whitening_transform(PhiX, tol, return_svd=True)
    # Compute reduced matrix:
    R = usv[2].T @ PhiY.T @ L
    # Diagonalize:
    d, W = eig(R)
    # Sort and filter eigenvalues:
    d, W = filter_ev(d, W) 
    d = d[-nev:]
    W = W[:, -nev:]
    # Transform eigenvectors to full basis:
    W = L @ W
    # Apply to data:
    Wdata = W.T @ PhiX

    return d, W, Wdata.T

def filter_ev(d, W, eps1=-np.inf, eps2=1.01, eps_im=np.inf):
    """
    Sort and filter eigenvalues for the Koopman operator. Sorts eigenvalues
    in ascending order by magnitude of the real part. Also, remove all
    eigenvalues exceeding 1.0 + eps.

    Parameters:
    ----------
    d, array(n):        eigenvalues
    W, array(n, r)      asscoiated eigenvectors
    eps1, eps2, float   discard all eigenvalues with real part smaller than eps1
                        or greater than eps2
    eps_im, float:      discard all eigenvalues with imaginary part greater than
                        eps_im

    Return:
    ------
    d, W,               sorted eigenvalues and eigenvectors
    """
    # Sort in ascending order by magnitude of the real part:\
    ind = np.argsort(np.real(d))
    d = d[ind]
    W = W[:, ind]
    # Filter out spurious ev larger than one plus eps:
    ind = np.where(np.logical_and(np.real(d) > eps1, 
                    np.logical_and(np.real(d) < eps2, np.abs(np.imag(d)) < eps_im)))[0]
    
    d = d[ind]
    W = W[:, ind]

    return d, W

def whitening_transform(PhiX, tol, rmin=0, return_svd=False, complex_val=False):
    """ Compute whitening transformation of the mass matrix based on truncated
        singular value decomposition.

      Parameters:
     -----------
     PhiX, array(n, m):     functional (lifted) time series, i.e. evalutation of n
                            basis functions at m data points.
                            NOTE: In Koopman operator estimation, PhiX only comprise the
                            first m-t time steps, where t is the lag time.
     tol, float             relative truncation threshold for SVD
     rmin, int:             minimal rank to be retained when truncating singular values
     return_V, bool:        return matrix of right singular vectors of the data matrix

    Returns:
    --------
    L, array(n, r)          linear transformation to reduced basis obtained from SVD
    V, array(m, r)          reduced right singular vectors, if return_V==True.
    """
    # Compute SVD:
    U, s, V = svd(PhiX, full_matrices=False)

    # Determine truncation rank and shrink SVD components:
    ind = np.where(s / s[0] >= tol)[0]
    r = np.maximum(ind.shape[0], rmin)
    U = U[:, :r]
    s = s[:r]

    if complex_val:
        V = V[:r, :].conj().T
    else:
        V = V[:r, :].T

    # Compute whitening transformation:
    L = U * (s**(-1))[None, :]

    if return_svd:
        return L, (U, s, V)
    else:
        return L

def sample_rff_gaussian(d, p, sigma):
    """ Draw a sample from the spectral density for a Gaussian kernel.
    Parameters:
    -----------
    d, int:         dimension of the state space for Gaussian kernel
    p, int:         number of samples from spectral density
    sigma, float:   bandwidth of the Gaussian kernel

    Returns:
    --------
    Omega (d, p):   p samples drawn from d-dimensional spectral density

    """

    return (1.0 / sigma) * np.random.randn(d, p)

def sample_rff_gauss_periodic(d, p, L, sigma, kmax):
    """ Draw a sample from the spectral density for a periodic Gaussian kernel.
    Parameters:
    -----------
    d, int:         dimension of the state space for Gaussian kernel
    p, int:         number of samples from spectral density
    L, float:       half-period of the domain, i.e. the domain is [-L, L]^d.
    sigma, float:   bandwidth of the Gaussian kernel
    kmax, int:      maximal wavenumber; random frequencies are drawn from the
                    hyper-grid $omega_0 * [-kmax, kmax]^d$.

    Returns:
    --------
    Omega (d, p):   p samples drawn from d-dimensional spectral density

    """
    # Fundamental frequency:
    omega0 = np.pi / L
    # Admissible wavenumbers in one dimension:
    kvec = range(-kmax, kmax + 1)
    # Compute probabilities in one dimension:
    pvec = np.zeros((2 * kmax + 1,))
    for kk in range(-kmax, kmax + 1):
        pvec[kk + kmax] = iv(kk, sigma ** (-2)) / np.exp(sigma ** (-2))
    pvec /= np.sum(pvec)
    # Prepare output:
    Omega = np.zeros((d, p))
    # Generate random frequencies from product distribution:
    for jj in range(d):
        Omega[jj, :] = omega0 * np.random.choice(kvec, size=p, replace=True, p=pvec)

    return Omega

def split_by_lag(Xfull, lag):
    """
    Parameters:
    -----------
    Xfull, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    lag, int:       lag time for Koopman estimation

    Returns:
    --------
    X, Y (d, M):    time-shifted concatenated data arrays Each row of
                    Y is the time-shited data point for the same row in X.
    """
    if isinstance(Xfull, list):
        X = np.zeros((Xfull[0].shape[0], 0))
        Y = np.zeros((Xfull[0].shape[0], 0))
        for Xii in Xfull:
            X = np.concatenate((X, Xii[:, :-lag]), axis=1)
            Y = np.concatenate((Y, Xii[:, lag:]), axis=1)
    else:
        X = Xfull[:, :-lag]
        Y = Xfull[:, lag:]

    return X, Y
