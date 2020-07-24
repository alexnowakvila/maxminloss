import cvxopt as cvx
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

def softmax(a, b):
    c = b * np.exp(a - a.max())
    return c / c.sum()

###############################################################################
# SPMP Python
###############################################################################


def maxmin_spmp_multiclass_p(nu, p, scores, Loss, max_iter, eta, Logs=None):

    n_states = Loss.shape[0]
    scores = np.expand_dims(scores, 1)
    nu = np.expand_dims(nu, 1)
    p = np.expand_dims(p, 1)

    mu_avg = np.zeros((n_states, 1))
    q_avg = np.zeros((n_states, 1))

    nu_avg = np.zeros((n_states, 1))
    p_avg = np.zeros((n_states, 1))
    dual_gaps = []
    energies = []
    if Logs == None: Logs = [max_iter - 1]
    for k in range(max_iter):
        q = softmax(-eta * np.dot(Loss, nu), p)
        mu = softmax(+eta * np.dot(Loss.T, p) + eta * scores, nu)
        if np.isnan(mu.max()):
            import pdb; pdb.set_trace()
        # p = softmax(-eta * np.dot(Loss, mu), q)
        p = softmax(-eta * np.dot(Loss, mu), p)
        # nu = softmax(eta * np.dot(Loss.T, q) + eta * scores, mu)
        nu = softmax(eta * np.dot(Loss.T, q) + eta * scores, nu)
        q_avg = k * q_avg / (k+1) + q / (k+1) 
        mu_avg = k * mu_avg / (k+1) + mu / (k+1)

        p_avg = k * p_avg / (k+1) + p / (k+1) 
        nu_avg = k * nu_avg / (k+1) + nu / (k+1)
        if k in Logs:
            m1 = np.max(np.dot(Loss.T, q_avg) + scores)
            m2 = np.min(np.dot(Loss, mu_avg)) + np.dot(scores.T, mu_avg)
            dual_gap = (m1 - m2).item()
            dual_gaps.append(dual_gap)
            en = np.dot(q_avg.T, np.dot(Loss, mu_avg) 
                        + np.dot(scores.T, mu_avg))
            energies.append(en.item())
    mu_avg = mu_avg.ravel()
    q_avg = q_avg.ravel()
    return mu_avg, q_avg, nu_avg.ravel(), p_avg.ravel(), dual_gaps, energies


###############################################################################
# CVXOPT Python
###############################################################################


def maxmin_multiclass_cvxopt(scores, Loss):
    solvers.options['show_progress'] = False
    n_states = Loss.shape[0]
    Loss = matrix(Loss)
    Al = np.ones((1, n_states))
    Ar = np.zeros((1, 1))
    A = matrix(np.concatenate((Al, Ar), axis=1))
    b = matrix(np.ones((1)))
    Gl = -1 * np.eye(n_states)
    Gl = np.concatenate((-1 * Loss, Gl), axis=0)
    Gr = np.concatenate((np.ones((n_states, 1)), np.zeros((n_states, 1))), axis=0)
    G = matrix(np.concatenate((Gl, Gr), axis=1))
    h = matrix(np.zeros((2 * n_states)))
    c = matrix(np.concatenate((-scores, -np.ones(1)), axis=0))
    sol=solvers.lp(c,G, h, A, b)
    en = -1 * sol['primal objective']
    dual_gap = sol['gap']
    mu = np.array(sol['x'][:n_states]).flatten()
    return mu, en, dual_gap, sol['x'][-1]


if __name__ == "__main__":
    n_states = 10
    Loss = np.ones((n_states, n_states))
    np.fill_diagonal(Loss, 0.0)
    scores = np.random.random_sample((n_states,))

    L = np.max(Loss) * np.log(n_states)
    eta =  np.log(n_states) / (2 * L)

    # initialize variables
    nu = np.ones(n_states) / n_states
    p = np.ones(n_states) / n_states
    max_iter = 10000
    Logs = list(np.arange(max_iter))
    mu_p, q_avg, dg, En_p = maxmin_spmp_multiclass_p(nu, p, scores,
                                Loss, max_iter, eta, Logs=Logs)