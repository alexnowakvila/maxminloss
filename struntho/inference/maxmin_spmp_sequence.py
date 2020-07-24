import sys
# sys.path.append("..")
import cvxopt as cvx
from cvxopt import matrix, solvers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import scipy.special as sp

from struntho.inference.sum_product_chain import sum_product_p
from struntho.inference._sum_product_chain import viterbi, sum_product_c

def softmax(a, b):
    c = b * np.exp(a)
    return c / c.sum(1, keepdims=True)

def maxmin_spmp_sequence_p(nu_nodes,
                            nu_edges,
                            p,
                            unary_potentials,
                            pairwise_potentials,
                            Loss,
                            max_iter,
                            eta,
                            sum_product_cython=False):
    """
        INPUT 

        unary_potentials: length * n_states
        pairwise_potentials: n_states * n_states  (pwpot same at all edges)
        edges: (length - 1) * 2
        L: n_states * n_states

        OUPTUT

        node_marginals: length * n_states
        pairwise_marginals: (length - 1) * n_states * n_states
    """

    # choose sum product functionality
    sum_product = sum_product_c if sum_product_cython else sum_product_p

    def grad_entropy(MU, edges):
        marginal_nodes, marginal_edges = MU
        grad_nodes = np.log(marginal_nodes + 1e-5)
        grad_edges = -np.log(marginal_edges + 1e-5)
        return grad_nodes, grad_edges

    n_states = pairwise_potentials.shape[0]
    length = unary_potentials.shape[0]
    # initialize optimization variables
    
    nu_nodes = np.log(nu_nodes + 1e-16)
    nu_edges = np.log(nu_edges + 1e-16)

    # initialize auxiliar variables
    q = np.zeros((length, n_states)) 
    mu_nodes = np.zeros((length, n_states)) 
    mu_edges = np.zeros((length - 1, n_states, n_states))
    
    
    # initialize averages
    q_avg = np.zeros((length, n_states)) 
    mu_avg_nodes = np.zeros((length, n_states)) 
    mu_avg_edges = np.zeros((length - 1, n_states, n_states))
    p_avg = np.zeros((length, n_states)) 
    nu_avg_nodes = np.zeros((length, n_states)) 
    nu_avg_edges = np.zeros((length - 1, n_states, n_states))
    
    

    # repeated_potentials = np.tile(pairwise_potentials, length - 1)
    repeated_potentials = np.repeat(pairwise_potentials[np.newaxis, :, :], length - 1, axis=0)

    dual_gaps = []

    max_iter = length * max_iter
    for k in range(max_iter):

        # FIRST PROXIMAL MAPPING
        q = softmax(-eta * np.dot(np.exp(nu_nodes), Loss.T), p)

        # prepare uscores
        uscores = eta * np.dot(p, Loss) + eta * unary_potentials - nu_nodes
        uscores[0] = uscores[0] + nu_nodes[0]
        uscores[-1] = uscores[-1] + nu_nodes[-1]
        bscores = eta * repeated_potentials + nu_edges
        sum_product(uscores, bscores, mu_nodes, mu_edges)

        # SECOND PROXIMAL MAPPING
        p = softmax(-eta * np.dot(np.exp(mu_nodes), Loss.T), p)
        # prepare uscores
        uscores = eta * np.dot(q, Loss) + eta * unary_potentials - nu_nodes
        uscores[0] = uscores[0] + nu_nodes[0]
        uscores[-1] = uscores[-1] + nu_nodes[-1]
        bscores = eta * repeated_potentials + nu_edges
        sum_product(uscores, bscores, nu_nodes, nu_edges)

        # UPDATE AVERAGES
        q_avg = k * q_avg / (k+1) + q / (k+1) 
        mu_avg_nodes = k * mu_avg_nodes / (k+1) + np.exp(mu_nodes) / (k+1) 
        mu_avg_edges = k * mu_avg_edges / (k+1) + np.exp(mu_edges) / (k+1) 
        p_avg = k * p_avg / (k+1) + p / (k+1) 
        nu_avg_nodes = k * nu_avg_nodes / (k+1) + np.exp(nu_nodes) / (k+1) 
        nu_avg_edges = k * nu_avg_edges / (k+1) + np.exp(nu_edges) / (k+1) 

        # COMPUTE DUAL GAP
        ymax = np.zeros(length, dtype=np.int32)
        viterbi(np.dot(q_avg, Loss) + unary_potentials, pairwise_potentials, ymax)
        # print("ymax", ymax)
        #make one hot encoding
        node_embeddings = np.zeros((length, n_states), dtype=np.int)
        gx = np.ogrid[:length]
        node_embeddings[gx, ymax] = 1
        ##accumulated pairwise
        edges = np.stack((np.arange(0, length - 1), np.arange(1, length)), 1)
        sum_edge_embeddings = np.dot(node_embeddings[edges[:, 0]].T,
                    node_embeddings[edges[:, 1]])
        # compute value of y_max
        m1 = (np.dot(q_avg, Loss) + unary_potentials)[np.arange(length), ymax].sum()
        m2 = (pairwise_potentials * sum_edge_embeddings).sum()
        maxval = m1 + m2

        en1 = (unary_potentials * mu_avg_nodes).sum()
        en2 = (repeated_potentials * mu_avg_edges).sum()
        minval = np.min(np.dot(mu_avg_nodes, Loss), axis=1).sum() + en1 + en2
        dual_gap = maxval - minval
        # print("Iteration: {}. Dual gap: {}".format(k, dual_gap))
        dual_gaps.append(dual_gap)
        # check for positive values
        # if mu_nodes.max() > 0.1: pdb.set_trace()
    out1 = [[mu_avg_nodes, mu_avg_edges], q_avg]
    out2 = [[nu_avg_nodes, nu_avg_edges], p_avg]
    return out1, out2, dual_gaps


def maxmin_spmp_sequence_p2(nu_nodes,
                            nu_edges,
                            p,
                            unary_potentials,
                            pairwise_potentials,
                            Loss,
                            max_iter,
                            eta,
                            sum_product_cython=False):
    """
        INPUT 

        unary_potentials: length * n_states
        pairwise_potentials: n_states * n_states  (pwpot same at all edges)
        edges: (length - 1) * 2
        L: n_states * n_states

        OUPTUT

        node_marginals: length * n_states
        pairwise_marginals: (length - 1) * n_states * n_states
    """

    # choose sum product functionality
    sum_product = sum_product_c if sum_product_cython else sum_product_p

    def grad_entropy(MU, edges):
        marginal_nodes, marginal_edges = MU
        grad_nodes = np.log(marginal_nodes + 1e-10)
        grad_edges = -np.log(marginal_edges + 1e-10)
        return grad_nodes, grad_edges

    n_states = pairwise_potentials.shape[0]
    length = unary_potentials.shape[0]
    # initialize optimization variables
    
    nu_nodes = np.log(nu_nodes + 1e-16)
    nu_edges = np.log(nu_edges + 1e-16)

    # initialize auxiliar variables
    q = np.zeros((length, n_states)) 
    mu_nodes = np.zeros((length, n_states)) 
    mu_edges = np.zeros((length - 1, n_states, n_states))
    
    # initialize averages
    q_avg = np.zeros((length, n_states)) 
    mu_avg_nodes = np.zeros((length, n_states)) 
    mu_avg_edges = np.zeros((length - 1, n_states, n_states))
    p_avg = np.zeros((length, n_states)) 
    nu_avg_nodes = np.zeros((length, n_states)) 
    nu_avg_edges = np.zeros((length - 1, n_states, n_states))

    # repeated_potentials = np.tile(pairwise_potentials, length - 1)
    repeated_potentials = np.repeat(pairwise_potentials[np.newaxis, :, :], length - 1, axis=0)

    dual_gaps = []

    max_iter = length * max_iter
    for k in range(max_iter):

        # FIRST PROXIMAL MAPPING
        q = softmax(-eta * np.dot(np.exp(nu_nodes), Loss.T), p)

        # prepare uscores
        uscores = eta * np.dot(p, Loss) + eta * unary_potentials - nu_nodes
        uscores[0] = uscores[0] + nu_nodes[0]
        uscores[-1] = uscores[-1] + nu_nodes[-1]
        bscores = eta * repeated_potentials + nu_edges
        sum_product(uscores, bscores, mu_nodes, mu_edges)

        # SECOND PROXIMAL MAPPING
        # if np.exp(mu_nodes).max() == np.inf or mu_nodes.min() == -np.inf:
        #     import pdb; pdb.set_trace()
        p = softmax(-eta * np.dot(np.exp(mu_nodes), Loss.T), q)
        # prepare uscores
        uscores = eta * np.dot(q, Loss) + eta * unary_potentials - mu_nodes
        uscores[0] = uscores[0] + mu_nodes[0]
        uscores[-1] = uscores[-1] + mu_nodes[-1]
        bscores = eta * repeated_potentials + mu_edges
        # if np.isinf(uscores.max()):
        #     import pdb; pdb.set_trace()
        sum_product(uscores, bscores, nu_nodes, nu_edges)
        

        # UPDATE AVERAGES
        q_avg = k * q_avg / (k+1) + q / (k+1) 
        mu_avg_nodes = k * mu_avg_nodes / (k+1) + np.exp(mu_nodes) / (k+1) 
        mu_avg_edges = k * mu_avg_edges / (k+1) + np.exp(mu_edges) / (k+1) 
        p_avg = k * p_avg / (k+1) + p / (k+1) 
        nu_avg_nodes = k * nu_avg_nodes / (k+1) + np.exp(nu_nodes) / (k+1) 
        nu_avg_edges = k * nu_avg_edges / (k+1) + np.exp(nu_edges) / (k+1) 

        # COMPUTE DUAL GAP
        ymax = np.zeros(length, dtype=np.int32)
        viterbi(np.dot(q_avg, Loss) + unary_potentials, pairwise_potentials, ymax)
        # print("ymax", ymax)
        #make one hot encoding
        node_embeddings = np.zeros((length, n_states), dtype=np.int)
        gx = np.ogrid[:length]
        node_embeddings[gx, ymax] = 1
        ##accumulated pairwise
        edges = np.stack((np.arange(0, length - 1), np.arange(1, length)), 1)
        sum_edge_embeddings = np.dot(node_embeddings[edges[:, 0]].T,
                    node_embeddings[edges[:, 1]])
        # compute value of y_max
        m1 = (np.dot(q_avg, Loss) + unary_potentials)[np.arange(length), ymax].sum()
        m2 = (pairwise_potentials * sum_edge_embeddings).sum()
        maxval = m1 + m2

        en1 = (unary_potentials * mu_avg_nodes).sum()
        en2 = (repeated_potentials * mu_avg_edges).sum()
        minval = np.min(np.dot(mu_avg_nodes, Loss), axis=1).sum() + en1 + en2
        dual_gap = maxval - minval
        # print("Iteration: {}. Dual gap: {}".format(k, dual_gap))
        dual_gaps.append(dual_gap)
        # check for positive values
        # if mu_nodes.max() > 0.1: pdb.set_trace()
    out1 = [[mu_avg_nodes, mu_avg_edges], q_avg]
    out2 = [[nu_avg_nodes, nu_avg_edges], p_avg]
    return out1, out2, dual_gaps


# def CVXOPT(unary_potentials, pairwise_potentials, Loss):

#     Loss = matrix(Loss)
#     n_states = pairwise_potentials.shape[0]
#     length = unary_potentials.shape[0]

#     # COMPUTE MATRIX A

#     A1 = np.zeros((n_states, n_states ** 2))
#     A2 = np.tile(np.arange(n_states), (n_states, n_states))
#     for j in range(n_states):
#         A1[j, j * n_states: (j+1) * n_states] = 1.
#         A2[j] = -1 * (A2[j] % n_states == j).astype(float)
#     A3 = np.concatenate((A1, A2), axis=1)
#     A = np.zeros((n_states * (length - 2), (length - 1) * n_states ** 2))
#     for l in range(length - 2):
#         A[l * n_states: (l+1) * n_states, l * n_states ** 2: (l+2) * n_states ** 2] = A3
#     # A has shape n_states * (length - 1) X length * n_states ** 2
#     # A4 = np.zeros((length - 1, (length - 1) * n_states ** 2))
#     # for l in range(length - 1):
#     #     A4[l, l * n_states ** 2 : (l+1) * n_states ** 2] = 1.
#     # A = np.concatenate((A, A4), axis=0)
#     # # insert part associated to z
#     # A = np.concatenate((A, np.zeros((A.shape[0], length))), axis=1)
#     # assert A.shape[0] == (length - 2) * n_states + length - 1

#     A4 = np.zeros((1, (length - 1) * n_states ** 2))
#     A4[0,:n_states ** 2] = 1.
#     A = np.concatenate((A, A4), axis=0)
#     # insert part associated to z
#     A = np.concatenate((A, np.zeros((A.shape[0], length))), axis=1)
#     assert A.shape[0] == (length - 2) * n_states + 1

#     # COMPUTE VECTOR b

#     b = np.zeros(A.shape[0])
#     b[(length - 2) * n_states:] = 1.

#     # COMPUTE MATRIX G
#     # we separate the computation between G1, G2, G3, G4
    
#     G0 = np.zeros((n_states, n_states ** 2))
#     for j in range(n_states):
#         g = np.ones((n_states, 1)).dot(Loss[[j], :])
#         G0[j] = g.flatten()
#     G1 = np.zeros((length * n_states, (length - 1) * n_states ** 2))
#     for l in range(length - 1):
#         G1[l * n_states: (l+1) * n_states, l * n_states ** 2: (l+1) * n_states ** 2] = G0
#     G1[(length - 1) * n_states : length * n_states, (length - 2) * n_states ** 2 : (length - 1) * n_states ** 2] = G0
#     G1 = -1 * G1
#     G2 = np.zeros((length * n_states, length))
#     for l in range(length):
#         G2[l * n_states: (l+1)* n_states, l] = 1.
#     G3 = -1 * np.eye((length - 1) * n_states ** 2)
#     G4 = np.zeros(((length - 1) * n_states ** 2, length))
#     G = np.concatenate((G1, G2), axis=1)
#     G = np.concatenate((G, np.concatenate((G3, G4), axis=1)), axis=0)

#     # COMPUTE VECTOR h
#     h = np.zeros(G.shape[0])

#     # COMPUTE COST VECTOR c
#     C1 = np.tile(pairwise_potentials.flatten(), (length - 1, 1))
#     for l in range(length - 1):
#         C1[l] += unary_potentials[[l], :].transpose().dot(np.ones((1, n_states))).flatten()
#     C1[length - 2] += unary_potentials[[length - 1], :].transpose().dot(np.ones((1, n_states))).flatten()
    
#     # C1 has shape length - 1 X n_states ** 2
#     C2 = np.ones(length)
#     c = -1 * np.concatenate((C1.flatten(), C2), axis=0)

#     # PRINT SHAPES
#     print("c has shape (length - 1) * n_states * n_states + length = {}".format(c.shape[0]))
#     print("G has shape (length * n_states + (length - 1) * n_states * n_states) X ((length - 1) * n_states * n_states + length) = {}".format(G.shape))
#     print("h has shape length * n_states + (length - 1) * n_states * n_states = {}".format(h.shape[0]))
#     print("A has shape ((length - 2) * n_states + (length - 1)) X ((length - 1) * n_states * n_states + length) = {}".format(A.shape))
#     print("b has shape (length - 2) * n_states + (length - 1) = {}".format(b.shape[0]))
#     assert c.shape[0] == (length - 1) * n_states * n_states + length
#     assert G.shape[0] == (length * n_states + (length - 1) * n_states * n_states)
#     assert G.shape[1] == ((length - 1) * n_states * n_states + length)
#     assert h.shape[0] == length * n_states + (length - 1) * n_states * n_states
#     # assert A.shape[0] == ((length - 2) * n_states + (length - 1))
#     assert A.shape[0] == ((length - 2) * n_states + 1)
#     assert A.shape[1] == ((length - 1) * n_states * n_states + length)
#     # assert b.shape[0] == (length - 2) * n_states + (length - 1)
#     assert b.shape[0] == (length - 2) * n_states + 1
    
#     # check rank
#     # print(A.shape[0] - np.linalg.matrix_rank(A))
#     # pass to matrix format
#     c, G, h, A, b = matrix(c), matrix(G), matrix(h), matrix(A), matrix(b)
#     sol=solvers.lp(c, G, h, A, b)
#     en = -1 * sol['primal objective']
#     dual_gap = sol['gap']

#     mu = np.array(sol['x'][: (length - 1) * n_states ** 2]).flatten()
#     mu_edges = np.reshape(mu, (length - 1, n_states, n_states))
    
#     mu_nodes = np.zeros((length, n_states))
#     for l in range(length - 1):
#         mu_nodes[l] = mu_edges[l].sum(0)
#     mu_nodes[length - 1] = mu_edges[-1].sum(1)
#     out = [[mu_nodes, mu_edges], None]
#     return out, en, dual_gap


if __name__ == "__main__":
    np.random.seed(1)

    eps = 1e-3
    n_states = 5
    Loss = np.ones((n_states, n_states))
    np.fill_diagonal(Loss, 0.0)
    Loss = toeplitz(np.arange(n_states))
    length = 10

    unary_potentials = np.random.random_sample((length, n_states))
    pairwise_potentials = np.random.random_sample((n_states, n_states))
    edges = np.stack((np.arange(0, length - 1), np.arange(1, length)), 1)
    p = np.ones((length, n_states)) / n_states
    nu_nodes = np.ones((length, n_states)) / n_states
    nu_edges = np.ones((length - 1, n_states, n_states)) / (n_states ** 2)
    max_iter = 50
    eta = 1 / (2 * np.max(Loss))

    out, dual_gaps = maxmin_spmp_sequence_p(nu_nodes,
                                            nu_edges,
                                            p,
                                            unary_potentials,
                                            pairwise_potentials,
                                            Loss,
                                            max_iter,
                                            eta,
                                            sum_product_cython=True)
