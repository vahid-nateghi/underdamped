"""
Modified from deeptime, which can also be found on Github:
https://github.com/deeptime-ml/deeptime/blob/main/deeptime/markov/tools/analysis/dense/_pcca.py

originaly authored by: Susanna Roeblitz, Marcus Weber, Frank Noe
"""

import numpy as np

def _pcca_connected_isa(eigenvectors, n_clusters):
    """
    PCCA+ spectral clustering method using the inner simplex algorithm.

    Clusters the first n_cluster eigenvectors of a transition matrix in order to cluster the states.
    This function assumes that the state space is fully connected, i.e. the transition matrix whose
    eigenvectors are used is supposed to have only one eigenvalue 1, and the corresponding first
    eigenvector (evec[:,0]) must be constant.

    Parameters
    ----------
    eigenvectors : ndarray
        A matrix with the sorted eigenvectors in the columns. The stationary eigenvector should
        be first, then the one to the slowest relaxation process, etc.

    n_clusters : int
        Number of clusters to group to.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    rot_mat : ndarray (m x m)
        A rotation matrix that rotates the dominant eigenvectors to yield the PCCA memberships, i.e.:
        chi = np.dot(evec, rot_matrix

    References
    ----------
    [1] P. Deuflhard and M. Weber, Robust Perron cluster analysis in conformation dynamics.
        in: Linear Algebra Appl. 398C M. Dellnitz and S. Kirkland and M. Neumann and C. Schuette (Editors)
        Elsevier, New York, 2005, pp. 161-184

    """
    (n, m) = eigenvectors.shape

    # do we have enough eigenvectors?
    if n_clusters > m:
        raise ValueError("Cannot cluster the (" + str(n) + " x " + str(m)
                         + " eigenvector matrix to " + str(n_clusters) + " clusters.")

    # check if the first, and only the first eigenvector is constant
    diffs = np.abs(np.max(eigenvectors, axis=0) - np.min(eigenvectors, axis=0))
    assert diffs[0] < 1e-6, "First eigenvector is not constant. This indicates that the transition matrix " \
                            "is not connected or the eigenvectors are incorrectly sorted. Cannot do PCCA."
    assert diffs[1] > 1e-6, "An eigenvector after the first one is constant. " \
                            "Probably the eigenvectors are incorrectly sorted. Cannot do PCCA."

    # local copy of the eigenvectors
    c = eigenvectors[:, list(range(n_clusters))]

    ortho_sys = np.copy(c)
    max_dist = 0.0

    # representative states
    ind = np.zeros(n_clusters, dtype=np.int32)

    # select the first representative as the most outlying point
    for (i, row) in enumerate(c):
        if np.linalg.norm(row, 2) > max_dist:
            max_dist = np.linalg.norm(row, 2)
            ind[0] = i

    # translate coordinates to make the first representative the origin
    ortho_sys -= c[ind[0], None]

    # select the other m-1 representatives using a Gram-Schmidt orthogonalization
    for k in range(1, n_clusters):
        max_dist = 0.0
        temp = np.copy(ortho_sys[ind[k - 1]])

        # select next farthest point that is not yet a representative
        for (i, row) in enumerate(ortho_sys):
            row -= np.dot(np.dot(temp, np.transpose(row)), temp)
            distt = np.linalg.norm(row, 2)
            if distt > max_dist and i not in ind[0:k]:
                max_dist = distt
                ind[k] = i
        ortho_sys /= np.linalg.norm(ortho_sys[ind[k]], 2)

    # print "Final selection ", ind

    # obtain transformation matrix of eigenvectors to membership matrix
    rot_mat = np.linalg.inv(c[ind])
    # print "Rotation matrix \n ", rot_mat

    # compute membership matrix
    chi = np.dot(c, rot_mat)
    # print "chi \n ", chi

    return chi, rot_mat