cimport cython
from cython cimport floating
from struntho.utils._cython_blas cimport (_scal_memview,
                                        _axpy_memview,
                                        _gemm_memview)
from libc.math cimport exp, log

cpdef softmax1_c(floating[::1] a, floating[::1] b):
    # a := softmax(a, b)
    cdef:
        int i
        int n = a.shape[0]
        floating norm = 0.0
        floating themax = a[0]
    for i in range(n):
        themax = max(themax, a[i])
    for i in range(n):
        a[i] = b[i] * exp(a[i] - themax)
        norm += a[i]
    for i in range(n):
        a[i] /= norm
        
cpdef linear_comb_c(floating alpha, floating beta,
                    floating[::1] a, floating[::1] b):
    _scal_memview(alpha, a)
    _axpy_memview(beta, b, a)
    
cpdef floating max_c(floating[::1] a):
    cdef:
        int i
        int n = a.shape[0]
        floating themax = a[0]
    for i in range(n):
        themax = max(themax, a[i])
    return themax

cpdef floating min_c(floating[::1] a):
    cdef:
        int i
        int n = a.shape[0]
        floating themin = a[0]
    for i in range(n):
        themin = min(themin, a[i])
    return themin

cpdef floating logsumexp(floating[:] arr, floating themax):
    # logsumexp in the last dimension
    cdef:
        int incx = 1
        int i
        int n = arr.shape[0]
        floating out = 0.0
    for i in range(n):
        out += exp(arr[i] - themax) # + 1e-10
    # if out < 1e-10:
    #     out = 1e-10
    out = log(out) + themax
    return out

cpdef softmax2_c(floating[:, :] a, floating[:, :] b):
    # a := softmax(a, b)
    cdef:
        int i
        int j
        int m = a.shape[0]
        int n = a.shape[1]
        floating norm
    for j in range(m):
        norm = 0.0
        for i in range(n):
            a[j, i] = b[j, i] * exp(a[j, i])
            norm += a[j, i]
        for i in range(n):
            a[j, i] /= norm

cpdef apply_log2(floating[:, :] arr, int m, int n):
    cdef:
        int i
        int j
    for i in range(m):
        for j in range(n):
            arr[i, j] = log(arr[i, j] + 1e-10)
            
cpdef apply_log3(floating[:, :, :] arr, int m, int n, int l):
    cdef:
        int i
        int j
        int k
    for i in range(m):
        for j in range(n):
            for k in range(l):
                arr[i, j, k] = log(arr[i, j ,k] + 1e-10)
                
cpdef apply_exp2(floating[:, :] arr, int m, int n):
    cdef:
        int i
        int j
    for i in range(m):
        for j in range(n):
            arr[i, j] = exp(arr[i, j])
            
cpdef apply_exp3(floating[:, :, :] arr, int m, int n, int l):
    cdef:
        int i
        int j
        int k
    for i in range(m):
        for j in range(n):
            for k in range(l):
                arr[i, j, k] = exp(arr[i, j ,k])
                

cpdef augment_nodes(floating[:, :] uscores,
                   floating[:, :] p,
                   floating[:, :] Loss,
                   floating[:, :] unary_potentials,
                   floating[:, :] nu_nodes,
                   floating eta,
                   int length,
                   int n_states):
    """ 
        uscores = eta * np.dot(p, Loss) + eta * unary_potentials - nu_nodes
        uscores[0] = uscores[0] + nu_nodes[0]
        uscores[-1] = uscores[-1] + nu_nodes[-1]
    """
    cdef:
        int i
        int j
    _gemm_memview(0, 0, 1.0, p, Loss, 0.0, uscores)
    for i in range(length):
        for j in range(n_states):
            uscores[i, j] = eta * uscores[i, j] + eta * unary_potentials[i, j]
            if i > 0 and i < length - 1:
                uscores[i, j] = uscores[i, j] - nu_nodes[i, j]

    
cpdef augment_edges(floating[:, :, :] bscores,
                    floating[:, :] pairwise_potentials,
                    floating[:, :, :] nu_edges,
                    floating eta,
                    int length,
                    int n_states):
    """
    bscores = eta * repeated_potentials + nu_edges
    """
    cdef:
        int i
        int j
        int k
    for i in range(length - 1):
        for j in range(n_states):
            for k in range(n_states):
                bscores[i, j, k] = eta * pairwise_potentials[j, k] + nu_edges[i, j, k]


cpdef linear_comb2(int n, int m, floating alpha, floating beta,
                floating[:, :] a, floating[:, :] b, int expn):
    cdef:
        int i
        int j
    for i in range(n):
        for j in range(m):
            a[i, j] = a[i, j] * alpha
            if expn == 0:
                a[i, j] = a[i, j] + b[i, j] * beta
            else:
                a[i, j] = a[i, j] + exp(b[i, j]) * beta

cpdef linear_comb3(int l, int n, int m, floating alpha, floating beta,
                floating[:, :, :] a, floating[:, :, :] b, int expn):
    cdef:
        int i
        int j
        int k
    for k in range(l):
        for i in range(n):
            for j in range(m):
                a[k, i, j] = a[k, i, j] * alpha
                if expn == 0:
                    a[k, i, j] = a[k, i, j] + b[k, i, j] * beta
                else:
                    a[k, i, j] = a[k, i, j] + exp(b[k, i, j]) * beta