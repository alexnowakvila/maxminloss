import pytest
import time

import numpy as np

from scipy.linalg import toeplitz

from struntho.utils._testing import assert_allclose
from struntho.inference.maxmin_spmp_sequence import maxmin_spmp_sequence_p, maxmin_spmp_sequence_p2
from struntho.inference._maxmin_spmp_sequence import maxmin_spmp_sequence_c, maxmin_spmp_sequence_c2

import matplotlib
import matplotlib.pyplot as plt

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

def test_SPMP():
    length = 5
    n_states = 5

    Loss = np.ones((n_states, n_states))
    np.fill_diagonal(Loss, 0.0)
    # Loss = toeplitz(np.arange(n_states))

    unary_potentials = np.random.random_sample((length, n_states))
    pairwise_potentials = np.random.random_sample((n_states, n_states))
    edges = np.stack((np.arange(0, length - 1), np.arange(1, length)), 1)
    p = np.ones((length, n_states)) / n_states
    nu_nodes = np.ones((length, n_states)) / n_states
    nu_edges = np.ones((length - 1, n_states, n_states)) / (n_states ** 2)
    max_iter = 100
    eta = 1 / (2 * np.max(Loss))
    # eta = 5.
    start = time.time()
    out1_p, out2_p, dg_p = maxmin_spmp_sequence_p2(nu_nodes,
                                            nu_edges,
                                            p,
                                            unary_potentials,
                                            pairwise_potentials,
                                            Loss,
                                            max_iter,
                                            eta,
                                            sum_product_cython=True)
    slow = time.time() - start
    mun_p, mue_p, q_p = out1_p[0][0], out1_p[0][1], out1_p[1]
    nun_p, nue_p, p_p = out2_p[0][0], out2_p[0][1], out2_p[1]
    start = time.time()
    out1_c, out2_c, dg_c = maxmin_spmp_sequence_c2(nu_nodes,
                                            nu_edges,
                                            p,
                                            unary_potentials,
                                            pairwise_potentials,
                                            Loss,
                                            max_iter,
                                            eta)
    fast = time.time() - start
    mun_c, mue_c = np.array(out1_c[0][0]), np.array(out1_c[0][1])
    nun_c, nue_c = np.array(out2_c[0][0]), np.array(out2_c[0][1])
    q_c = np.array(out1_c[1])
    p_c = np.array(out2_c[1])
    dg_c = np.array(dg_c)
    assert_allclose(mun_c, mun_p, rtol=7)
    assert_allclose(mue_c, mue_p, rtol=7)
    assert_allclose(q_c, q_p, rtol=7)
    assert_allclose(nun_c, nun_p, rtol=7)
    assert_allclose(nue_c, nue_p, rtol=7)
    assert_allclose(p_c, p_p, rtol=7)
    assert_allclose(dg_c, dg_p, rtol=7)
    # print("errors:", np.square(mun_c-mun_p).sum(), np.square(mue_c-mue_p).sum(), np.square(q_c-q_p).sum(), np.square(dg_c-dg_p).sum())

    # print("improvement is {}".format(slow / fast))

    # plt.figure()
    # plt.plot(dg_c, label='cython')
    # plt.plot(dg_p, label='python')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    test_SPMP()