import cython
from cython cimport floating
# from utils._cython_inference cimport logsumexp
from struntho.utils._cython_blas cimport (_dot_memview,
                                _copy_memview,
                                _axpy_memview,
                                _gemv_memview,
                                _gemm_memview)
from struntho.utils._cython_inference cimport (
                                    softmax2_c,
                                    apply_log2,
                                    apply_log3,
                                    apply_exp2,
                                    apply_exp3,
                                    augment_nodes,
                                    augment_edges,
                                    linear_comb2,
                                    linear_comb3)
from struntho.inference._sum_product_chain cimport sum_product_c, viterbi
from libc.math cimport exp, log
cimport numpy as np
import numpy as np 

def maxmin_spmp_sequence_c2(floating[:, :] nu_nodes,
                            floating[:, :, :] nu_edges,
                            floating[:, :] p,
                            floating[:, :] unary_potentials,
                            floating[:, :] pairwise_potentials,
                            floating[:, :] Loss,
                            int max_iter,
                            floating eta):
    
    # define typed variables
    cdef:
        # auxuliar pair of variables
        floating[:, :] q
        floating[:, :] mu_nodes
        floating[:, :, :] mu_edges
        # average pair of variables
        floating[:, :] q_avg
        floating[:, :] mu_avg_nodes
        floating[:, :, :] mu_avg_edges
        floating[:, :] p_avg
        floating[:, :] nu_avg_nodes
        floating[:, :, :] nu_avg_edges
        # vector of dual gaps
        floating[::1] dual_gaps
        # auxiliar vector to compute the dual gap
        floating[:, :] aux
        floating[:, :] aux2
        # score variables
        floating[:, :] uscores
        floating[:, :, :] bscores
        int[::1] ymax

    cdef:
        # indexes used
        int i
        int k
        int j
        int r
        floating k_f = 0.0
        # size of the instance
        int n_states = Loss.shape[0]
        int length = unary_potentials.shape[0]
        # logging
        floating alpha
        floating beta
        floating dual_gap
        floating primal = 0.0
        floating dual = 0.0
        # others
        floating themin

    # assert n_states == pairwise_potentials.shape[0]
    max_iter = length * max_iter
    # pass to logarithmic parametrization
    apply_log2(nu_nodes, length, n_states)
    apply_log3(nu_edges, length - 1, n_states, n_states)

    # initialize auxiliar variables
    q = np.zeros((length, n_states)) 
    mu_nodes = np.zeros((length, n_states)) 
    mu_edges = np.zeros((length - 1, n_states, n_states))
    ymax = np.empty(length, dtype=np.int32)

    # initialize averages
    q_avg = np.zeros((length, n_states)) 
    mu_avg_nodes = np.zeros((length, n_states)) 
    mu_avg_edges = np.zeros((length - 1, n_states, n_states))
    p_avg = np.zeros((length, n_states)) 
    nu_avg_nodes = np.zeros((length, n_states)) 
    nu_avg_edges = np.zeros((length - 1, n_states, n_states))

    # initialize scores
    uscores = np.zeros((length, n_states)) 
    bscores = np.zeros((length - 1, n_states, n_states))
    
    aux = np.zeros((length, n_states))
    aux2 = np.zeros((length, n_states))
    dual_gaps = np.empty(max_iter)

    #main loop
    for k in range(max_iter):
        #######################################################################
        # FIRST PROXIMAL MAPPING
        #######################################################################
        
        # update mu
        # usc:= eta * (p.dot(L) + 0.0  * usc) + eta * unary_potentials - log(nu_nodes)
        augment_nodes(uscores, p, Loss, unary_potentials,
                        nu_nodes, eta, length, n_states)
        # bsc := eta * pairwise_potentials + log(nu_edges)
        augment_edges(bscores, pairwise_potentials,
                        nu_edges, eta, length, n_states)            
        # (mu_nodes, mu_edges) := sum product(usc, bsc)
        sum_product_c(uscores, bscores, mu_nodes, mu_edges)  # sum product
        # now q, p are in normal format
        
        # update q 
        # nu_nodes := exp(nu_nodes)
        apply_exp2(nu_nodes, length, n_states)
        # q:= -eta * nu_nodes.dot(L.T) + 0.0 * q
        # blas_dot_sum_matrix(nu_nodes, Loss, q, -eta, 0.0, 0, 1) 
        _gemm_memview(0, 1, -eta, nu_nodes, Loss, 0.0, q)
        # q:= softmax(q, p, axis=1)
        # blas_softmax(q, p, length, n_states)
        softmax2_c(q, p)
        # now p, q and nu_nodes are in normal format

        
        #######################################################################
        # SECOND PROXIMAL MAPPING
        #######################################################################
        
        # update nu
        # usc:= eta * q.dot(L) + eta * unary_potentials - log(mu_nodes)
        augment_nodes(uscores, q, Loss, unary_potentials,
                            mu_nodes, eta, length, n_states)
        # bsc := eta * pairwise_potentials + log(mu_edges)
        augment_edges(bscores, pairwise_potentials,
                            mu_edges, eta, length, n_states)
        # (nu_nodes, nu_edges) := sum product(usc, bsc)
        sum_product_c(uscores, bscores, nu_nodes, nu_edges)
        # now p, q are in normal format
        
        # update p
        # mu_nodes := exp(mu_nodes)
        apply_exp2(mu_nodes, length, n_states)
        # p:= -eta * mu_nodes.dot(L.T)
        # blas_dot_sum_matrix(mu_nodes, Loss, p, -eta, 0.0, 0, 0)
        _gemm_memview(0, 0, -eta, mu_nodes, Loss, 0.0, p)

        # p:= softmax(p, q, axis=1)
        # blas_softmax(p, q, length, n_states)
        softmax2_c(p, q)
        # now p, q and mu_nodes are in normal format
        
        #######################################################################
        # UPDATE AVERAGES
        #######################################################################
        
        alpha = k_f / (k_f + 1.0)
        beta = 1. / (k_f + 1.0)
        # q_avg = k * q_avg / (k+1) + q / (k+1) 
        linear_comb2(length, n_states, alpha, beta, q_avg, q, 0)  
        # mu_avg_nodes = k * mu_avg_nodes / (k+1) + mu_nodes / (k+1) 
        linear_comb2(length, n_states, alpha, beta, mu_avg_nodes, mu_nodes, 0)
        # mu_avg_edges = k * mu_avg_edges / (k+1) + mu_edges / (k+1) 
        linear_comb3(length - 1, n_states, n_states, alpha, beta,
                            mu_avg_edges, mu_edges, 1)
        linear_comb2(length, n_states, alpha, beta, p_avg, p, 0) 
        linear_comb2(length, n_states, alpha, beta, nu_avg_nodes, nu_nodes, 1)
        linear_comb3(length - 1, n_states, n_states, alpha, beta,
                            nu_avg_edges, nu_edges, 1)
        
        #######################################################################
        # COMPUTE DUAL GAP
        #######################################################################
        
        ########################### primal ####################################
        
        # apply viterbi to compute ymax
        for i in range(length):
            for j in range(n_states):
                aux[i, j] = unary_potentials[i, j]
        # _copy_memview(unary_potentials, aux)
        # blas_dot_sum_matrix(q_avg, Loss, aux, 1.0, 1.0, 0, 0)
        _gemm_memview(0, 0, 1.0, q_avg, Loss, 1.0, aux)
        for i in range(length):
            for j in range(n_states):
                aux2[i, j] = aux[i, j]
        viterbi(aux, pairwise_potentials, ymax)  # ymax = viterbi(np.dot(q_avg, Loss) + unary_potentials, pairwise_potentials)
        # compute value of y_max
        for i in range(length):
            primal += aux2[i, ymax[i]]
        for i in range(length - 1):
            primal += pairwise_potentials[ymax[i], ymax[i + 1]]
            
        ############################### dual ##################################
        
#         en1 = (unary_potentials * mu_avg_nodes).sum()
#         en2 = (repeated_potentials * mu_avg_edges).sum()
#         minval = np.min(np.dot(mu_avg_nodes, Loss), axis=1).sum() + en1 + en2
        
        for i in range(length):
            for j in range(n_states):
                dual += unary_potentials[i, j] * mu_avg_nodes[i, j]
        for i in range(length - 1):
            for j in range(n_states):
                for r in range(n_states):
                    dual += pairwise_potentials[j, r] * mu_avg_edges[i, j, r]
                    
        _gemm_memview(0, 1, 1.0, mu_avg_nodes, Loss, 0.0, aux2)
        for i in range(length):
            themin = aux2[i, 0]
            for j in range(n_states):
                themin = min(aux2[i, j], themin)
            dual += themin
        
        # compute difference
        dual_gap = primal - dual
        primal = 0.0
        dual = 0.0
        dual_gaps[k] = dual_gap
        k_f += 1
    out1 = [[mu_avg_nodes, mu_avg_edges], q_avg]
    out2 = [[nu_avg_nodes, nu_avg_edges], p_avg]
    return out1, out2, dual_gaps



def maxmin_spmp_sequence_c(floating[:, :] nu_nodes,
                            floating[:, :, :] nu_edges,
                            floating[:, :] p,
                            floating[:, :] unary_potentials,
                            floating[:, :] pairwise_potentials,
                            floating[:, :] Loss,
                            int max_iter,
                            floating eta):
    
    # define typed variables
    cdef:
        # auxuliar pair of variables
        floating[:, :] q
        floating[:, :] mu_nodes
        floating[:, :, :] mu_edges
        # average pair of variables
        floating[:, :] q_avg
        floating[:, :] mu_avg_nodes
        floating[:, :, :] mu_avg_edges
        floating[:, :] p_avg
        floating[:, :] nu_avg_nodes
        floating[:, :, :] nu_avg_edges
        # vector of dual gaps
        floating[::1] dual_gaps
        # auxiliar vector to compute the dual gap
        floating[:, :] aux
        floating[:, :] aux2
        # score variables
        floating[:, :] uscores
        floating[:, :, :] bscores
        int[::1] ymax

    cdef:
        # indexes used
        int i
        int k
        int j
        int r
        floating k_f = 0.0
        # size of the instance
        int n_states = Loss.shape[0]
        int length = unary_potentials.shape[0]
        # logging
        floating alpha
        floating beta
        floating dual_gap
        floating primal = 0.0
        floating dual = 0.0
        # others
        floating themin

    # assert n_states == pairwise_potentials.shape[0]
    max_iter = length * max_iter
    # pass to logarithmic parametrization
    apply_log2(nu_nodes, length, n_states)
    apply_log3(nu_edges, length - 1, n_states, n_states)

    # initialize auxiliar variables
    q = np.zeros((length, n_states)) 
    mu_nodes = np.zeros((length, n_states)) 
    mu_edges = np.zeros((length - 1, n_states, n_states))
    ymax = np.empty(length, dtype=np.int32)

    # initialize averages
    q_avg = np.zeros((length, n_states)) 
    mu_avg_nodes = np.zeros((length, n_states)) 
    mu_avg_edges = np.zeros((length - 1, n_states, n_states))
    p_avg = np.zeros((length, n_states)) 
    nu_avg_nodes = np.zeros((length, n_states)) 
    nu_avg_edges = np.zeros((length - 1, n_states, n_states))

    # initialize scores
    uscores = np.zeros((length, n_states)) 
    bscores = np.zeros((length - 1, n_states, n_states))
    
    aux = np.zeros((length, n_states))
    aux2 = np.zeros((length, n_states))
    dual_gaps = np.empty(max_iter)

    #main loop
    for k in range(max_iter):
        #######################################################################
        # FIRST PROXIMAL MAPPING
        #######################################################################
        
        # update mu
        # usc:= eta * (p.dot(L) + 0.0  * usc) + eta * unary_potentials - log(nu_nodes)
        augment_nodes(uscores, p, Loss, unary_potentials,
                        nu_nodes, eta, length, n_states)
        # bsc := eta * pairwise_potentials + log(nu_edges)
        augment_edges(bscores, pairwise_potentials,
                        nu_edges, eta, length, n_states)            
        # (mu_nodes, mu_edges) := sum product(usc, bsc)
        sum_product_c(uscores, bscores, mu_nodes, mu_edges)  # sum product
        # now q, p are in normal format
        
        # update q 
        # nu_nodes := exp(nu_nodes)
        apply_exp2(nu_nodes, length, n_states)
        # q:= -eta * nu_nodes.dot(L.T) + 0.0 * q
        # blas_dot_sum_matrix(nu_nodes, Loss, q, -eta, 0.0, 0, 1) 
        _gemm_memview(0, 1, -eta, nu_nodes, Loss, 0.0, q)
        # q:= softmax(q, p, axis=1)
        # blas_softmax(q, p, length, n_states)
        softmax2_c(q, p)
        # now p, q and nu_nodes are in normal format

        
        #######################################################################
        # SECOND PROXIMAL MAPPING
        #######################################################################
        
        # update nu
        # usc:= eta * q.dot(L) + eta * unary_potentials - log(mu_nodes)
        apply_log2(nu_nodes, length, n_states)
        augment_nodes(uscores, q, Loss, unary_potentials,
                            nu_nodes, eta, length, n_states)
        # bsc := eta * pairwise_potentials + log(mu_edges)
        augment_edges(bscores, pairwise_potentials,
                            nu_edges, eta, length, n_states)
        # (nu_nodes, nu_edges) := sum product(usc, bsc)
        sum_product_c(uscores, bscores, nu_nodes, nu_edges)
        # now p, q are in normal format
        
        # update p
        # mu_nodes := exp(mu_nodes)
        apply_exp2(mu_nodes, length, n_states)
        # p:= -eta * mu_nodes.dot(L.T)
        for i in range(length):
            for j in range(n_states):
                aux[i, j] = p[i, j]
        # blas_dot_sum_matrix(mu_nodes, Loss, p, -eta, 0.0, 0, 0)
        _gemm_memview(0, 0, -eta, mu_nodes, Loss, 0.0, p)

        # p:= softmax(p, q, axis=1)
        # blas_softmax(p, q, length, n_states)
        softmax2_c(p, aux)
        # now p, q and mu_nodes are in normal format
        
        #######################################################################
        # UPDATE AVERAGES
        #######################################################################
        
        alpha = k_f / (k_f + 1.0)
        beta = 1. / (k_f + 1.0)
        # q_avg = k * q_avg / (k+1) + q / (k+1) 
        linear_comb2(length, n_states, alpha, beta, q_avg, q, 0)  
        # mu_avg_nodes = k * mu_avg_nodes / (k+1) + mu_nodes / (k+1) 
        linear_comb2(length, n_states, alpha, beta, mu_avg_nodes, mu_nodes, 0)
        # mu_avg_edges = k * mu_avg_edges / (k+1) + mu_edges / (k+1) 
        linear_comb3(length - 1, n_states, n_states, alpha, beta,
                            mu_avg_edges, mu_edges, 1)
        linear_comb2(length, n_states, alpha, beta, p_avg, p, 0) 
        linear_comb2(length, n_states, alpha, beta, nu_avg_nodes, nu_nodes, 1)
        linear_comb3(length - 1, n_states, n_states, alpha, beta,
                            nu_avg_edges, nu_edges, 1)
        
        #######################################################################
        # COMPUTE DUAL GAP
        #######################################################################
        
        ########################### primal ####################################
        
        # apply viterbi to compute ymax
        for i in range(length):
            for j in range(n_states):
                aux[i, j] = unary_potentials[i, j]
        # _copy_memview(unary_potentials, aux)
        # blas_dot_sum_matrix(q_avg, Loss, aux, 1.0, 1.0, 0, 0)
        _gemm_memview(0, 0, 1.0, q_avg, Loss, 1.0, aux)
        for i in range(length):
            for j in range(n_states):
                aux2[i, j] = aux[i, j]
        viterbi(aux, pairwise_potentials, ymax)  # ymax = viterbi(np.dot(q_avg, Loss) + unary_potentials, pairwise_potentials)
        # compute value of y_max
        for i in range(length):
            primal += aux2[i, ymax[i]]
        for i in range(length - 1):
            primal += pairwise_potentials[ymax[i], ymax[i + 1]]
            
        ############################### dual ##################################
        
#         en1 = (unary_potentials * mu_avg_nodes).sum()
#         en2 = (repeated_potentials * mu_avg_edges).sum()
#         minval = np.min(np.dot(mu_avg_nodes, Loss), axis=1).sum() + en1 + en2
        
        for i in range(length):
            for j in range(n_states):
                dual += unary_potentials[i, j] * mu_avg_nodes[i, j]
        for i in range(length - 1):
            for j in range(n_states):
                for r in range(n_states):
                    dual += pairwise_potentials[j, r] * mu_avg_edges[i, j, r]
                    
        _gemm_memview(0, 1, 1.0, mu_avg_nodes, Loss, 0.0, aux2)
        for i in range(length):
            themin = aux2[i, 0]
            for j in range(n_states):
                themin = min(aux2[i, j], themin)
            dual += themin
        
        # compute difference
        dual_gap = primal - dual
        primal = 0.0
        dual = 0.0
        dual_gaps[k] = dual_gap
        k_f += 1
    out1 = [[mu_avg_nodes, mu_avg_edges], q_avg]
    out2 = [[nu_avg_nodes, nu_avg_edges], p_avg]
    return out1, out2, dual_gaps