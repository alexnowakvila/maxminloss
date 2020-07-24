import test

import numpy as np

from struntho.utils._testing import assert_allclose
from struntho.inference.sum_product_chain import sum_product_p
from struntho.inference._sum_product_chain import sum_product_c

import time
np.random.seed(2)


def test_sum_product_chain():
    eps = 1e-3
    n_states = 10
    length = 20

    unary_scores = 10e4 * (np.random.random_sample((length, n_states)) - 0.5)
    pairwise_scores = 10e4 * (np.random.random_sample((length - 1, n_states, n_states)) - 0.5)
    start = time.time()
    n_p = np.empty([length, n_states])
    e_p = np.empty([length - 1, n_states, n_states])
    sum_product_p(unary_scores, pairwise_scores, n_p, e_p)
    slow = time.time() - start
    start = time.time()

    n_c = np.empty([length, n_states])
    e_c = np.empty([length - 1, n_states, n_states])
    sum_product_c(unary_scores, pairwise_scores, n_c, e_c)
    fast = time.time() - start
    
    # import pdb; pdb.set_trace()
    assert_allclose(n_c, n_p, rtol=7)
    assert_allclose(e_c, e_p, rtol=7)
    # assert_allclose(bm_c, bm_p, rtol=7)
    # assert_allclose(fm_c, fm_p, rtol=7)
    # assert_allclose(logp_c, logp_p, rtol=7)
    print("Speedup is {}".format( slow / fast ))

