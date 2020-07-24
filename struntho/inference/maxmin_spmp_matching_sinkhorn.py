import cvxopt as cvx
from cvxopt import matrix, solvers
import numpy as np
import ot  # optimal transport library
from scipy.optimize import linear_sum_assignment  # hungarian algorithm
import matplotlib.pyplot as plt
import pdb


def sinkhorn_knopp(a, b, M, reg, u=None, v=None, max_iter=1000,
                   verbose=False, **kwargs):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    # if Logs == None: Logs = [max_iter - 1]
    dual_gaps = []

    # initalize dual variables
    if u is None and v is None:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    tmp2 = np.empty(b.shape, dtype=M.dtype)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    for cpt in range(max_iter):
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        # if cpt in Logs:
        #     # primal
        #     P = u.reshape((-1, 1)) * K * v.reshape((1, -1))
        #     primal = (P * M).sum() + reg * (P * (np.log(P + 1e-8) - 1)).sum()
        #     # dual
        #     dual = np.expand_dims(u, 0).dot(K.dot(np.expand_dims(v, 1))).item()
        #     dual = (np.log(u) * a).sum() + (np.log(v) * b).sum() - dual
        #     dual = dual * reg
        #     print("primal is {} dual is {}".format(primal, dual))
        #     # dual gap
        #     # print(u, v)
        #     dual_gaps.append(primal - dual)
        cpt = cpt + 1
    return u.reshape((-1, 1)) * K * v.reshape((1, -1)), u, v  #, dual_gaps


###############################################################################
# SPMP Python
###############################################################################


def maxmin_spmp_matching_sinkhorn(P, Q, S, max_iter, eta,
                                    sink_warmstart=False,
				    sink_iter=10,
                                    Logs=None):
    # min_Q max_P  min_U max_V
    # (Q, P) and (U, V)
    P_avg = np.zeros_like(P)
    Q_avg = np.zeros_like(Q)
    U_avg = np.zeros_like(P)
    V_avg = np.zeros_like(Q)
    dual_gaps = []
    energies = []
    n_states = P.shape[0]
    a, b = np.ones(n_states), np.ones(n_states)
    if Logs == None: Logs = [max_iter - 1]
    if sink_warmstart:
        Uu = np.ones(n_states) / n_states
        Uv = np.ones(n_states) / n_states
        Vu = np.ones(n_states) / n_states
        Vv = np.ones(n_states) / n_states
        Qu = np.ones(n_states) / n_states
        Qv = np.ones(n_states) / n_states
        Pu = np.ones(n_states) / n_states
        Pv = np.ones(n_states) / n_states
    else:
        Uu, Uv, Vu, Vv, Qu, Qv, Pu, Pv = [None] * 8
    for k in range(max_iter):
        U, Uu, Uv = sinkhorn_knopp(a, b, -eta * P - np.log(Q + 1e-6), 1, u=Uu, v=Uv, max_iter=sink_iter)
        V, Vu, Vv = sinkhorn_knopp(a, b, -eta * (S - Q) - np.log(P + 1e-6), 1, u=Vu, v=Vv, max_iter=sink_iter)
        Q, Qu, Qv = sinkhorn_knopp(a, b, -eta * V - np.log(Q + 1e-6), 1, u=Qu, v=Qv, max_iter=sink_iter)
        P, Pu, Pv = sinkhorn_knopp(a, b, -eta * (S - U) - np.log(P + 1e-6), 1, u=Pu, v=Pv, max_iter=sink_iter)

        P_avg = k * P_avg / (k+1) + P / (k+1)
        Q_avg = k * Q_avg / (k+1) + Q / (k+1) 
        U_avg = k * U_avg / (k+1) + U / (k+1)
        V_avg = k * V_avg / (k+1) + V / (k+1) 
        if k in Logs:
            # compute energy and dual gap
            # compute primal
            cost = -1 * Q_avg + S
            row_ind, col_ind = linear_sum_assignment(-1 * cost)
            primal = cost[row_ind, col_ind].sum()
            # compute dual
            cost = -1 * P_avg
            row_ind, col_ind = linear_sum_assignment(cost)
            dual = cost[row_ind, col_ind].sum() + (P_avg * S).sum()
            # dual_gap
            dual_gap = primal - dual
            dual_gaps.append(dual_gap)
            # compute energy
            en = (- P_avg * Q_avg).sum() + (P_avg * S).sum()
            energies.append(en)
    return Q_avg, P_avg, U_avg, V_avg, dual_gaps, energies



if __name__ == "__main__":
    ###########################################################################
    # Sinkhorn Knopp
    ###########################################################################

    # n_states = 10
    # max_iter = 100
    # reg = 0.01
    # Logs = list(np.arange(0, max_iter, 1))
    # a = np.ones(n_states) / n_states
    # b = np.ones(n_states) / n_states
    # M = np.random.random((n_states, n_states))
    # OT1, dgs = sinkhorn_knopp(a, b, M, reg, max_iter=10000, Logs=Logs)
    # OT2 = sinkhorn_knopp(a, b, M, reg)[0]
    # assert np.sum((OT1 - OT2) ** 2) < 1e-7
    # print(OT1, OT2)
    # pdb.set_trace()
    # plt.figure(1)
    # plt.style.use(['seaborn-darkgrid'])
    # plt.plot(dgs)
    # plt.show()

    ###########################################################################
    # Max-Min oracle
    ###########################################################################

    n_states = 3
    max_iter = 20
    Logs = list(np.arange(0, max_iter, 1))

    P = np.ones((n_states, n_states)) / n_states
    Q = np.ones((n_states, n_states)) / n_states
    S = 10 * np.random.random((n_states, n_states))
    S = np.array([[1, 3, 4], [0, 2, 1], [5, 2, 9]])
    eta = 1.0

    Q_avg, P_avg, dgs1, ens = maxmin_spmp_matching_sinkhorn(P, Q, S,
                                                    max_iter, eta, warmstart=False, Logs=Logs)
    Q_avg, P_avg, dgs2, ens = maxmin_spmp_matching_sinkhorn(P, Q, S,
                                                    max_iter, eta, warmstart=True, Logs=Logs)

    plt.figure(1)
    plt.style.use(['seaborn-darkgrid'])
    plt.plot(dgs1)
    plt.plot(dgs2)
    plt.show()
