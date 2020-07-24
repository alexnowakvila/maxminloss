from struntho.utils._testing import assert_allclose

import numpy as np

from struntho.utils._cython_inference import softmax1_c
from struntho.utils._cython_inference import linear_comb_c
from struntho.utils._cython_inference import max_c
from struntho.utils._cython_inference import min_c
from struntho.utils._cython_inference import logsumexp
from struntho.utils._cython_inference import softmax2_c
from struntho.utils._cython_inference import apply_log2
from struntho.utils._cython_inference import apply_log3
from struntho.utils._cython_inference import apply_exp2
from struntho.utils._cython_inference import apply_exp3
from struntho.utils._cython_inference import augment_nodes
from struntho.utils._cython_inference import augment_edges
from struntho.utils._cython_inference import linear_comb2


RTOL = {np.float32: 1e-6, np.float64: 1e-12}
dtype = np.float64

def test_softmax1():
    rng = np.random.RandomState(0)
    a = rng.random_sample(10).astype(dtype, copy=False)
    b = rng.random_sample(10).astype(dtype, copy=False)
    expected = b * np.exp(a)
    expected = expected / expected.sum()
    actual = a.copy()
    softmax1_c(actual, b)
    assert_allclose(actual, expected, rtol=RTOL[dtype])
    
def test_linear_comb_memview():
    rng = np.random.RandomState(0)
    a = rng.random_sample(10).astype(dtype, copy=False)
    b = rng.random_sample(10).astype(dtype, copy=False)
    alpha = rng.random_sample()
    beta = rng.random_sample()
    expected = alpha * a + beta * b
    actual = a.copy()
    linear_comb_c(alpha, beta, actual, b)
    assert_allclose(actual, expected, rtol=RTOL[dtype])
    
def test_max_memview():
    rng = np.random.RandomState(0)
    a = rng.random_sample(10).astype(dtype, copy=False)
    expected = np.max(a)
    actual = max_c(a)
    assert_allclose(actual, expected, rtol=RTOL[dtype])
    
def test_min_memview():
    rng = np.random.RandomState(0)
    a = rng.random_sample(10).astype(dtype, copy=False)
    expected = np.min(a)
    actual = min_c(a)
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_logsumexp():
    rng = np.random.RandomState(0)
    a = rng.random_sample(10).astype(dtype, copy=False)
    expected = np.log( (np.exp(a - a.max())).sum()  ) + a.max()
    actual = logsumexp(a, a.max())
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_softmax2():
    rng = np.random.RandomState(0)
    a = rng.random_sample((10, 10)).astype(dtype, copy=False)
    b = rng.random_sample((10, 10)).astype(dtype, copy=False)
    expected = b * np.exp(a)
    expected = expected / expected.sum(1, keepdims=True)
    actual = a.copy()
    softmax2_c(actual, b)
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_apply_log2():
    rng = np.random.RandomState(0)
    a = rng.random_sample((10, 10)).astype(dtype, copy=False)
    expected = np.log(a + 1e-10)
    actual = a.copy()
    apply_log2(actual, 10, 10)
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_apply_log3():
    rng = np.random.RandomState(0)
    a = rng.random_sample((10, 10, 10)).astype(dtype, copy=False)
    expected = np.log(a + 1e-10)
    actual = a.copy()
    apply_log3(actual, 10, 10, 10)
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_apply_exp2():
    rng = np.random.RandomState(0)
    a = rng.random_sample((10, 10)).astype(dtype, copy=False)
    expected = np.exp(a)
    actual = a.copy()
    apply_exp2(actual, 10, 10)
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_apply_exp3():
    rng = np.random.RandomState(0)
    a = rng.random_sample((10, 10, 10)).astype(dtype, copy=False)
    expected = np.exp(a)
    actual = a.copy()
    apply_exp3(actual, 10, 10, 10)
    assert_allclose(actual, expected, rtol=RTOL[dtype])

def test_augment_nodes():
    rng = np.random.RandomState(0)
    uscores = rng.random_sample((10, 10)).astype(dtype, copy=False)
    p = rng.random_sample((10, 10)).astype(dtype, copy=False)
    Loss = rng.random_sample((10, 10)).astype(dtype, copy=False)
    unary_potentials = rng.random_sample((10, 10)).astype(dtype, copy=False)
    nu_nodes = rng.random_sample((10, 10)).astype(dtype, copy=False)
    eta = 0.1
    # expected
    expected = eta * np.dot(p, Loss) + eta * unary_potentials - nu_nodes
    expected[0] = expected[0] + nu_nodes[0]
    expected[-1] = expected[-1] + nu_nodes[-1]
    # actual
    augment_nodes(uscores, p, Loss, unary_potentials, nu_nodes, eta, 10, 10)
    assert_allclose(uscores, expected, rtol=RTOL[dtype])

def test_augment_edges():
    rng = np.random.RandomState(0)
    bscores = rng.random_sample((9, 10, 10)).astype(dtype, copy=False)
    pairwise_potentials = rng.random_sample((10, 10)).astype(dtype, copy=False)
    nu_edges = rng.random_sample((9, 10, 10)).astype(dtype, copy=False)
    eta = 0.1
    # expected
    repeated_potentials = np.repeat(pairwise_potentials[np.newaxis, :, :], 9, axis=0)
    expected = eta * repeated_potentials + nu_edges
    # actual
    augment_edges(bscores, pairwise_potentials, nu_edges, eta, 10, 10)
    assert_allclose(bscores, expected, rtol=RTOL[dtype])

def test_linear_comb2():
    rng = np.random.RandomState(0)
    a = rng.random_sample((10, 10)).astype(dtype, copy=False)
    b = rng.random_sample((10, 10)).astype(dtype, copy=False)
    alpha = 0.1
    beta = 0.9
    for expn in [0, 1]:
        if expn:
            expected = alpha * a + beta * np.exp(b)
        else:
            expected = alpha * a + beta * b
        actual = a.copy()
        linear_comb2(10, 10, alpha, beta, actual, b, expn)
        assert_allclose(actual, expected, rtol=RTOL[dtype])
