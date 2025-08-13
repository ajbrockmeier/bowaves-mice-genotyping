"""Sign-invariant and Shift-invariant k-means"""

import sys
import warnings

import numpy as np
from scipy.cluster.vq import vq


from sklearn.utils.extmath import stable_cumsum, squared_norm, row_norms
from sklearn.exceptions import ConvergenceWarning
from BOWaves.utilities import sikmeans_utils

from si_vq import si_vq,si2_vq


def _random_init(X, n_clusters, centroid_length, rng):
    n_samples = X.shape[0]
    seeds = rng.permutation(n_samples)[:n_clusters]
    centroids = X[seeds][:,:centroid_length]
#    centroids = sikmeans_utils.pick_random_windows(centroids, 1, centroid_length,
#                                                   rng).squeeze() # this is unnecessary as it will simply pick first centroid_length

    return centroids



###############################################################################
# Main algorithm

def si_kmeans(X, n_clusters, centroid_length, metric='cosine',
                               init='random', use_sign_invariant=False,do_sphere=True,
              n_init=10, max_iter=300, tol=1e-4, rng=None, verbose=False):
    """
    Shift-invariant k-means algorithm

    Parameters
    ----------
    X (numpy.ndarray):
        Data matrix with samples in its rows.
    n_clusters (int):
        Number of clusters to form, as well as the number of centroids to find.
    centroid_length (int):
        The length of each centroid.
    metric ('euclidean' or 'cosine'):
        Metric used to compute the distance between samples and cluster centroids. Default: 'euclidean'.
    init ('k-means++', 'random', numpy.ndarray, or a function):
        Method for initialization. If it's a function, it should have this
        call signature:
        centroids, shifts = init(
             X, n_clusters, centroid_length, rng, **kwargs).
        rng must be a Generator instance.
    n_init (int):
        The number of times the algorithm is run with different centroid seeds.
        The final results would be from the iteration where the inertia is the
        lowest.
    max_iter (init):
        Maximum number of iterations the algorithm will be run.
    tol (float):
        Upper bound that the squared euclidean norm of the change in the
        centroids must achieve to declare convergence.
    rng (int, Generator instance or None):
        Determines random number generation for centroid initialization. Use an
        int to make the randomness deterministic.
    verbose (bool):
        If True, print details about each iteration.

    Returns
    -------
    centroids (numpy.ndarray):
        A matrix with the learned centroids in its rows.
    labels (numpy.ndarray):
        labels[i] is the index of the centroid (row of `centroids`) closest
        to the sample X[i].
    shifts (numpy.ndarray):
        shift[i] is the shift that minimizes the distance to the closest
        centroid to the sample X[i].
    distances (numpy.ndarray):
        distances[i] is the distance from X[i,shift[i]:shift[i]+centroid_length]
        to its closest centroid.
    inertia (float):
        The sum of squared euclidean distances to the closest centroid of all the
        training samples.
    best_n_iter (int):
        Number of iterations needed to achieve convergence, according to `tol`.
    """

    rng = sikmeans_utils.check_rng(rng)

    best_labels, best_shifts, best_centroids = None, None, None
    best_distances, best_inertia, best_n_iter = None, None, None

    # subtract of mean of x for more accurate distance computations
    # NOTE: Can't do that because each centroid is the average of windows from X
    # that were chosen at different starting times.

    ss = rng.bit_generator._seed_seq
    child_seeds = ss.spawn(n_init)
    streams = [np.random.default_rng(s) for s in child_seeds]
       
    for seed in streams:
        # run a shift-invariant k-means once
        centroids, labels, shifts, distances, inertia, n_iter_ = si_kmeans_single(
            X, n_clusters, centroid_length, metric=metric, use_sign_invariant=use_sign_invariant,do_sphere=do_sphere,
            init=init, max_iter=max_iter, tol=tol, rng=seed, verbose=verbose)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_centroids = centroids.copy()
            best_labels = labels.copy()
            best_shifts = shifts.copy()
            best_distances = distances
            best_inertia = inertia
            best_n_iter = n_iter_

    distinct_clusters = len(set(best_labels))

    if distinct_clusters < n_clusters:
        warnings.warn(
            "Number of distinct clusters ({}) found smaller than "
            "n_clusters ({}). Possibly due to duplicate points "
            "in X.".format(distinct_clusters, n_clusters), ConvergenceWarning,
            stacklevel=2
        )

    return best_centroids, best_labels, best_shifts, best_distances, best_inertia, best_n_iter


def si_kmeans_single(X, n_clusters, centroid_length, metric='euclidean', use_sign_invariant=False, do_sphere=False,
                     init='k-means++', max_iter=300, tol=1e-3, rng=None, verbose=False):
    """
    Single run of shift-invariant k-means
    """

    rng = sikmeans_utils.check_rng(rng)


    best_labels, best_shifts, best_centroids = None, None, None
    best_distances, best_inertia = None, None

    # Random init only

    centroids = _random_init(X, n_clusters, centroid_length,rng)

    
    #The below is Dr. B's additions from the Jupyter notebook.
    #I've added the update step function to the utils file.
    #Adding here to test before PR
    labels, shifts, distances, signs = _assignment_step(
        X, centroids, metric, use_sign_invariant)
    centroids = _init_centroids_update_step(
        X, centroid_length, n_clusters, labels, shifts, signs, do_sphere) # NEW

    if verbose:
        print('Initialization completed.')

    for iteration in range(max_iter):
        centroids_old = centroids.copy()
        labels, shifts, distances, signs = _assignment_step(X, centroids, metric, use_sign_invariant)
        centroids = _centroids_update_step(
            X, centroid_length, n_clusters, labels, shifts, signs, do_sphere)

        inertia = distances.mean()

        if verbose:
            print("Iteration %2d, inertia %.3f" % (iteration, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_shifts = shifts.copy()
            best_centroids = centroids.copy()
            best_distances = distances
            best_inertia = inertia

        centroid_change = squared_norm(centroids_old - centroids)/n_clusters/centroid_length
        #print(centroid_change, tol)
        if centroid_change <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "centroid changes %e within tolerance %e"
                      % (iteration, centroid_change, tol))
            break

    if centroid_change > 0:
        # rerun asingment step in case of non-convergence so that predicted
        # labels match cluster centers
        best_labels, best_shifts, best_distances, best_signs = _assignment_step(X, best_centroids, metric, use_sign_invariant)
        best_inertia = distances.mean()

    return best_centroids, best_labels, best_shifts, best_distances, best_inertia, iteration+1


def _assignment_step(X, centroids, metric, use_sign_invariant):
    """
    Find the index of the shifted centroid that is closest to each sample

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows of X are samples.
    centroids (numpy.ndarray):
        Centroids of the clusters.

    Returns
    -------
    labels (numpy.ndarray):
        centroids[labels[i]] is the centroid closest to sample X[i]
    shifts (numpy.ndarray):
        X[i, shifts[i]:shifts[i]+centroid_length] is the window in X[i]  closest to centroids[labels[i]].
    distances (numpy.ndarray):
        distances[i] is the distance of X[i, shifts[i]:shifts[i]+ centroid_length] to the closest centroid.
    """

    if use_sign_invariant:
        labels, shifts, distances,signs = si2_vq( X, centroids, metric)
    else:
        labels, shifts, distances = si_vq(X, centroids, metric)
        signs = np.ones_like(shifts)

    return labels, shifts, distances, signs


def _centroids_update_step(X, centroid_length, n_clusters, labels, shifts, signs, do_sphere=False):
    """
    Update the cluster centroids
    """

    centroids = np.zeros((n_clusters, centroid_length))

    for sample_id, sample in enumerate(X):
        cluster_id = labels[sample_id]
        shift = shifts[sample_id]
        if do_sphere:
            x_shift = signs[sample_id]*sample[shift:shift+centroid_length]
            centroids[cluster_id] += x_shift/np.sqrt(np.sum(x_shift**2))
            
        else:
            centroids[cluster_id] += signs[sample_id]*sample[shift:shift+centroid_length]

    # NOTE: Some clusters might be empty
    cluster_id, cluster_size = np.unique(labels, return_counts=True)
    centroids[cluster_id, :] /= cluster_size[:, np.newaxis]

    return centroids

def _init_centroids_update_step(X, centroid_length, n_clusters, labels, shifts, signs,do_sphere=False):
    """
    Update the cluster centroids
    """

    cluster_ids, _ = np.unique(labels, return_counts=True)
    centroids = np.zeros((n_clusters, centroid_length))
    n_samples, sample_length = X.shape
    # adjust the shifts such that after adjustment the median shift is
    max_shift = sample_length - centroid_length
    opt_shift = max_shift/2
    adjusts = np.zeros((n_clusters))
    for k in cluster_ids:
        shifts_k = shifts[labels==k]
        adjusts[k] = opt_shift-np.median(shifts_k)

    cluster_sizes = np.zeros((n_clusters,1))
    for sample_id, sample in enumerate(X):
        cluster_id = labels[sample_id]
        temp = shifts[sample_id]+adjusts[cluster_id]
        if temp >= 0 and temp <= max_shift:
            shift = np.floor(temp).astype(int)
            if do_sphere:
                x_shift = signs[sample_id]*sample[shift:shift+centroid_length]
                centroids[cluster_id] += x_shift/np.sqrt(np.sum(x_shift**2))        
            else:
                centroids[cluster_id] += signs[sample_id]*sample[shift:shift+centroid_length]
            
            cluster_sizes[cluster_id] += 1

    # NOTE: Some clusters might be empty drop them
    #centroids/= cluster_sizes
    for k in np.where(cluster_sizes==0)[0]:
        centroids[k,:] = 0
    for k in np.nonzero(cluster_sizes)[0]:
        centroids[k,:]/= cluster_sizes[k]

    return centroids
