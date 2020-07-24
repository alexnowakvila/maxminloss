cimport cython
from cython cimport floating
from struntho.utils._cython_blas cimport _copy_memview, _dot_memview, _scal_memview, _axpy_memview, _gemv_memview 
# from utils._cython_blas cimport _gemv_memview 
from struntho.utils._cython_inference cimport (
        softmax1_c,
        linear_comb_c,
        max_c,
        min_c
    )

from libc.math cimport exp, log
cimport numpy as np
import numpy as np

np.import_array()

def multiclass_oracle_c(floating[::1] nu,
                        floating[::1] p,
                        floating[::1] scores,
                        floating[:, :] Loss,
                        int max_iter,
                        floating eta):
    cdef:
        floating[::1] q
        floating[::1] mu
        floating[::1] q_avg
        floating[::1] mu_avg
        floating[::1] p_avg
        floating[::1] nu_avg
        floating[::1] dual_gaps
        # np.ndarray[ndim=1, dtype=np.float64_t] energies
        floating[::1] aux

    cdef:
        int n_states = Loss.shape[0]
        int k
        floating k_f = 0.0
        int j
        floating dual_gap
        # float en
        floating alpha
        floating beta
    
    mu = np.empty(n_states)
    q = np.empty(n_states)
    mu_avg = np.zeros(n_states)
    q_avg = np.zeros(n_states)
    nu_avg = np.zeros(n_states)
    p_avg = np.zeros(n_states)
    aux = np.empty(n_states)

    dual_gaps = np.empty(max_iter, dtype=np.float64)
    # energies = np.empty(max_iter, dtype=np.float64)

    for k in range(max_iter):

        
        # update q
        # q = softmax(-eta * np.dot(Loss, np.expand_dims(nu, 1)).flatten(), p)
        _gemv_memview(0, -eta, Loss, nu, 0.0, q)
        softmax1_c(q, p)
        
        
        # copy scores to mu (mu = scores)
        _copy_memview(scores, mu)
        
        # update mu
        # mu = softmax(+eta * np.dot(Loss.T, np.expand_dims(p, 1)).flatten() + eta * scores, nu)
        _gemv_memview(1, eta, Loss, p, eta, mu)
        softmax1_c(mu, nu)
        
        # update p
        # copy p to aux
        _copy_memview(p, aux)
        _gemv_memview(0, -eta, Loss, mu, 0.0, p)
        softmax1_c(p, aux)
        
        # copy nu to aux
        _copy_memview(nu, aux)
        # copy scores to nu (nu = scores)
        _copy_memview(scores, nu)
        
        # update nu
        # nu = softmax(eta * np.dot(Loss.T, np.expand_dims(q, 1)).flatten() + eta * scores, mu)
        _gemv_memview(1, eta, Loss, q, eta, nu)
        softmax1_c(nu, aux)
        
        alpha = k_f / (k_f + 1.0)
        beta = 1. / (k_f + 1.0)
        # update average q
        # q_avg = k * q_avg / (k+1) + q / (k+1)
        linear_comb_c(alpha, beta, q_avg, q)
        
        # update average mu
        # mu_avg = k * mu_avg / (k+1) + mu / (k+1)
        linear_comb_c(alpha, beta, mu_avg, mu)

        linear_comb_c(alpha, beta, p_avg, p)
        linear_comb_c(alpha, beta, nu_avg, nu)
        
        # compute dual gap
        dual_gap = 0.0
        # primal
        _copy_memview(scores, aux)
        _gemv_memview(1, 1.0, Loss, q_avg, 1.0, aux)

        dual_gap = max_c(aux)
        # dual
        _gemv_memview(0, 1.0, Loss, mu_avg, 0.0, aux)

        dual_gap -= min_c(aux) + _dot_memview(scores, mu_avg)
        dual_gaps[k] = dual_gap

        k_f += 1.0
    return mu_avg, q_avg, nu_avg, p_avg, dual_gaps