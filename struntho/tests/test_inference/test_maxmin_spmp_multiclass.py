import pytest

import numpy as np

from scipy.linalg import toeplitz

from struntho.utils._testing import assert_allclose
from struntho.inference.maxmin_spmp_multiclass import maxmin_multiclass_cvxopt, maxmin_spmp_multiclass_p
from struntho.inference._maxmin_spmp_multiclass import multiclass_oracle_c



def create_losses(n_states):
    Losses = []
    # 0-1 loss
    loss = np.ones((n_states, n_states))
    np.fill_diagonal(loss, 0.0)
    Losses.append(loss)
    # ordinal loss
    Losses.append(toeplitz(np.arange(n_states)))
    # random loss
    loss = np.random.random_sample((n_states, n_states))
    np.fill_diagonal(loss, 0.0)
    Losses.append(loss)
    return Losses

def test_multiclass_oracle_p():
    # N_states, Precisions = [2, 5, 10], [1, 2, 3]
    N_states, Precisions = [5], [1, 2]
    for n_states in N_states:
        Losses = create_losses(n_states)
        for Loss in Losses:
            scores = np.random.random_sample((n_states,))
            # run cvxopt
            mu_cx, en_cx, _ , _ = maxmin_multiclass_cvxopt(scores, Loss)

            L = np.max(Loss) * np.log(n_states)
            eta =  np.log(n_states) / (2 * L)

            # initialize variables
            nu = np.ones(n_states) / n_states
            p = np.ones(n_states) / n_states
            
            Eps = [1 / (10 ** precision) for precision in Precisions]
            Logs = [int(4 * L / eps) for eps in Eps]
            max_iter = Logs[-1]
            mu_p, q_avg, _, _, _, En_p = maxmin_spmp_multiclass_p(nu, p, scores,
                                        Loss, max_iter, eta, Logs=Logs)

            for i, en_p in enumerate(En_p):
                assert_allclose(en_p, en_cx, rtol=Precisions[i])

def test_multiclass_oracle_c():
    n_states = 10
    Losses = create_losses(n_states)
    for Loss in Losses:
        Loss = np.random.random_sample((n_states, n_states))
        scores = np.random.random_sample((n_states,))
        L = np.max(Loss) * np.log(n_states)
        eta =  np.log(n_states) / (2 * L)

        # initialize variables
        nu = np.ones(n_states) / n_states
        p = np.ones(n_states) / n_states
        
        eps = 1e-2
        max_iter = int(4 * L / eps)
        Logs = list(np.arange(0, max_iter))
        mu_p, q_p, nu_p, p_p, dual_gaps_p, _ = maxmin_spmp_multiclass_p(nu, p, scores, Loss, max_iter, eta, Logs=Logs)
        nu = np.ones(n_states) / n_states
        p = np.ones(n_states) / n_states

        mu_c, q_c, nu_c, p_c, dual_gaps_c = multiclass_oracle_c(nu, p, scores, Loss, max_iter, eta)
        mu_c, q_c = np.array(mu_c), np.array(q_c) 
        nu_c, p_c = np.array(nu_c), np.array(p_c)
        dual_gaps_c = np.array(dual_gaps_c)
        assert_allclose(mu_c, mu_p, rtol=7)
        assert_allclose(q_c, q_p, rtol=7)
        assert_allclose(nu_c, nu_p, rtol=7)
        assert_allclose(p_c, p_p, rtol=7)
        assert_allclose(dual_gaps_c, dual_gaps_p, rtol=7)
        assert dual_gaps_c[-1] > 0 and dual_gaps_c[-1] < eps