import numpy as np
import pdb

def sum_product_p(uscores, bscores, umargs, bmargs):
    """Apply the sum-product algorithm on a chain
    :param uscores: array T*K, (unary) scores on individual nodes
    :param bscores: array (T-1)*K*K, (binary) scores on the edges
    :return: log-marginals on nodes, log-marginals on edges, log-partition
    """

    def logsumexp(arr, axis=None):
        themax = np.max(arr, axis=axis, keepdims=True)
        out = np.sum(np.exp(arr - themax), axis=axis)
        out = themax.flatten() + np.log(out)
        return out

    # I keep track of the islog messages instead of the messages
    # This is more stable numerically
    # 
    length, nb_class = uscores.shape
    if length == 1:
        log_partition = logsumexp(uscores[0])
        umargs[0] = uscores[0] - log_partition
        bmargs = np.zeros([length - 1, nb_class, nb_class])
        return 0

    bm = np.zeros([length - 1, nb_class])  # backward_messages
    fm = np.zeros([length - 1, nb_class])  # forward_messages

    # backward pass
    bm[-1] = logsumexp(bscores[-1] + uscores[-1], axis=-1)
    for t in range(length - 3, -1, -1):
        bm[t] = logsumexp(bscores[t] + uscores[t + 1] + bm[t + 1], axis=-1)

    # we compute the log-partition and include it in the forward messages
    log_partition = logsumexp(bm[0] + uscores[0])

    # forward pass
    fm[0] = logsumexp(bscores[0].T + uscores[0] - log_partition, axis=-1)
    for t in range(1, length - 1):
        fm[t] = logsumexp(bscores[t].T + uscores[t] + fm[t - 1], axis=-1)

    # unary marginals
    # umargs = np.empty([length, nb_class])
    umargs[0] = uscores[0] + bm[0] - log_partition
    umargs[-1] = fm[-1] + uscores[-1]
    for t in range(1, length - 1):
        umargs[t] = fm[t - 1] + uscores[t] + bm[t]

    # binary marginals
    # bmargs = np.empty([length - 1, nb_class, nb_class])

    if length == 2:
        # pdb.set_trace()
        bmargs[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] - log_partition
    else:
        bmargs[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] + bm[1] - log_partition
        bmargs[-1] = fm[-2, :, np.newaxis] + uscores[-2, :, np.newaxis] + bscores[-1] + uscores[-1]
        for t in range(1, length - 2):
            bmargs[t] = fm[t - 1, :, np.newaxis] + uscores[t, :, np.newaxis] + bscores[t] + \
                        uscores[t + 1] + bm[t + 1]
    # pdb.set_trace()
    
    


def viterbi_p(score, trans_score, path):
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
    
    n_samples, n_states = score.shape[0], score.shape[1]
    backp = np.zeros((n_samples, n_states), dtype=np.intc)
    # Forward recursion. score is reused as the DP table.
    for i in range(1, n_samples):
        for k in range(n_states):
            maxind = 0
            maxval = -np.inf
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