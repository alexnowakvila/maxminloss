import cython
from cython cimport floating
from struntho.utils._cython_inference cimport logsumexp
from libc.math cimport exp, log
cimport numpy as np
import numpy as np

np.import_array()

cdef double NEGINF = -np.inf

########################################################################
# LOG-PARTITION
########################################################################

def log_partition_c(floating[:, :] uscores, floating[:, :, :] bscores):
    """Apply the sum-product algorithm on a chain
    :param uscores: array T*K, (unary) scores on individual nodes
    :param bscores: array (T-1)*K*K, (binary) scores on the edges
    :return: log-marginals on nodes, log-marginals on edges, log-partition
    """

    cdef:
        int n_states = uscores.shape[1]
        int t
        int j
        int r
        floating log_partition
        int length
        floating themax = 0.0
        
    cdef:
        floating[:, :] bm  # backward_messages
        floating[:, :] fm  # forward_messages
        floating[:, :] aux  # auxiliar matrix
        # floating[:, :] umargs  # unary marginals
        # floating[:, :, :] bmargs  # pairwise marginals
        
    length = uscores.shape[0]
    # umargs = np.empty([length, n_states])
    # bmargs = np.empty([length - 1, n_states, n_states])

    # if length == 1:
    #     log_partition = logsumexp(uscores[0], n_states)
    #     for j in range(n_states):
    #         umargs[0, j] = uscores[0, j] - log_partition
    #     return 0

    bm = np.zeros([length - 1, n_states])
    fm = np.zeros([length - 1, n_states])  # forward_messages
    aux = np.zeros([n_states, n_states]) # forward_messages
    
    ########################################################################
    # backward pass
    ########################################################################
    
    # end node
    for j in range(n_states):
        for r in range(n_states):
            aux[j, r] = bscores[-1, j, r] + uscores[-1, r]
            if r == 0: themax = aux[j, r]
            themax = max(themax, aux[j ,r]) 
        bm[-1, j] = logsumexp(aux[j], themax)
    
    # move to the front
    for t in range(length - 3, -1, -1):
        for j in range(n_states):
            for r in range(n_states):
                aux[j, r] = bscores[t, j, r] + uscores[t + 1, r] + bm[t + 1, r]
                if r == 0: themax = aux[j, r]
                themax = max(themax, aux[j ,r])
            bm[t, j] = logsumexp(aux[j], themax)
        

    #########################################################################
    # compute the log-partition and include it in the forward messages
    #########################################################################
    
    for r in range(n_states):
        aux[0, r] = bm[0, r] + uscores[0, r]
        if r == 0: themax = aux[0, r]
        themax = max(themax, aux[0, r])
    log_partition = logsumexp(aux[0], themax)
    return log_partition

########################################################################
# SUM PRODUCT
########################################################################

cpdef sum_product_c(floating[:, :] uscores, floating[:, :, :] bscores,
                    floating[:, :] umargs, floating[:, :, :] bmargs):
    """Apply the sum-product algorithm on a chain
    :param uscores: array T*K, (unary) scores on individual nodes
    :param bscores: array (T-1)*K*K, (binary) scores on the edges
    :return: log-marginals on nodes, log-marginals on edges, log-partition
    """

    cdef:
        int n_states = uscores.shape[1]
        int t
        int j
        int r
        floating log_partition
        int length
        floating themax = 0.0
        
    cdef:
        floating[:, :] bm  # backward_messages
        floating[:, :] fm  # forward_messages
        floating[:, :] aux  # auxiliar matrix
        # floating[:, :] umargs  # unary marginals
        # floating[:, :, :] bmargs  # pairwise marginals
        
    length = uscores.shape[0]
    # umargs = np.empty([length, n_states])
    # bmargs = np.empty([length - 1, n_states, n_states])

    if length == 1:
        log_partition = logsumexp(uscores[0], n_states)
        for j in range(n_states):
            umargs[0, j] = uscores[0, j] - log_partition
        # bmargs = np.zeros([length - 1, n_states, n_states])
        return 0

    bm = np.zeros([length - 1, n_states])
    fm = np.zeros([length - 1, n_states])  # forward_messages
    aux = np.zeros([n_states, n_states]) # forward_messages
    
    ########################################################################
    # backward pass
    ########################################################################
    
    # end node
    for j in range(n_states):
        for r in range(n_states):
            aux[j, r] = bscores[-1, j, r] + uscores[-1, r]
            if r == 0: themax = aux[j, r]
            themax = max(themax, aux[j ,r]) 
        bm[-1, j] = logsumexp(aux[j], themax)
    
    # move to the front
    for t in range(length - 3, -1, -1):
        for j in range(n_states):
            for r in range(n_states):
                aux[j, r] = bscores[t, j, r] + uscores[t + 1, r] + bm[t + 1, r]
                if r == 0: themax = aux[j, r]
                themax = max(themax, aux[j ,r])
            bm[t, j] = logsumexp(aux[j], themax)
        

    #########################################################################
    # compute the log-partition and include it in the forward messages
    #########################################################################
    
    for r in range(n_states):
        aux[0, r] = bm[0, r] + uscores[0, r]
        if r == 0: themax = aux[0, r]
        themax = max(themax, aux[0, r])
    log_partition = logsumexp(aux[0], themax)
    
    ########################################################################
    # forward pass
    ########################################################################
    
    # first node
    for j in range(n_states):
        for r in range(n_states):
            aux[j, r] = bscores[0, r, j] + uscores[0, r] - log_partition
            if r == 0: themax = aux[j, r]
            themax = max(themax, aux[j ,r])     
        fm[0, j] = logsumexp(aux[j], themax)
    
    # move to the back
    for t in range(1, length - 1):
        for j in range(n_states):
            for r in range(n_states):
                aux[j, r] = bscores[t, r, j] + uscores[t, r] + fm[t - 1, r]
                if r == 0: themax = aux[j, r]
                themax = max(themax, aux[j ,r])
            fm[t, j] = logsumexp(aux[j], themax)

    ########################################################################
    # compute unary marginals
    ########################################################################
    
    
    for j in range(n_states):
        umargs[0, j] = uscores[0, j] + bm[0, j] - log_partition
    for j in range(n_states):
        umargs[-1, j] = fm[-1, j] + uscores[-1, j]
    for t in range(1, length - 1):
        for j in range(n_states):
            umargs[t, j] = fm[t - 1, j] + uscores[t, j] + bm[t, j]

    ########################################################################
    # compute pairwise marginals
    ########################################################################

    if length == 2:
        for j in range(n_states):
            for r in range(n_states):
                bmargs[0, j, r] = uscores[0, j] + bscores[0, j, r] + uscores[1, r] - log_partition
    else:
        for j in range(n_states):
            for r in range(n_states):
                bmargs[0, j, r] = uscores[0, j] + bscores[0, j, r] + uscores[1, r] + bm[1, r] - log_partition
                bmargs[-1, j, r] = fm[-2, j] + uscores[-2, j] + bscores[-1, j, r] + uscores[-1, r]
        for t in range(1, length - 2):
            for j in range(n_states):
                for r in range(n_states):
                    bmargs[t, j, r] = fm[t - 1, j] + uscores[t, j] + bscores[t, j, r] + \
                                uscores[t + 1, r] + bm[t + 1, r]


cpdef viterbi(floating[:, :] score,
                floating[:, :] trans_score, int[::1] path):
    """First-order Viterbi algorithm.

    Parameters
    ----------
    score : array, shape = (n_samples, n_states)
        Scores per sample/class combination; in a linear model, X * w.T.
        May be overwritten.
    trans_score : array, shape = (n_samples, n_states, n_states), optional
        Scores per sample/transition combination.

    References
    ----------
    L. R. Rabiner (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proc. IEEE 77(2):257-286.
    """

    cdef int[:, :] backp
    cdef floating candidate, maxval, themax
    cdef int i, j, k, n_samples, n_states
    
    n_samples, n_states = score.shape[0], score.shape[1]
    backp = np.zeros((n_samples, n_states), dtype=np.intc)
    # Forward recursion. score is reused as the DP table.
    for i in range(1, n_samples):
        for k in range(n_states):
            maxind = 0
            maxval = NEGINF
            for j in range(n_states):
                candidate = score[i - 1, j] + score[i, k] + trans_score[j, k]
                # candidate = score[i - 1, j] + score[i, k] + trans_score[j, k]
                if candidate > maxval:
                    maxind = j
                    maxval = candidate
            score[i, k] = maxval
            backp[i, k] = maxind
    
    # Path backtracking
    # path = np.empty(n_samples, dtype=np.intc)
    
    themax = score[n_samples - 1, 0]
    # path[n_samples - 1] = score[n_samples - 1, :].argmax()
    path[n_samples - 1] = 0
    # compute the argmax
    for i in range(n_states):
        if themax < score[n_samples - 1, i]:
            path[n_samples - 1] = i
            themax = score[n_samples - 1, i]

    for i in range(n_samples - 2, -1, -1):
        path[i] = backp[i + 1, path[i + 1]]