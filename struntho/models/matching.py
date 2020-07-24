import numpy as np
import cvxopt as cvx
from cvxopt import matrix, solvers
import pdb
import scipy.special as sp
from struntho.utils._testing import assert_allclose
from struntho.inference.maxmin_spmp_matching_sinkhorn import maxmin_spmp_matching_sinkhorn as maxmin_spmp_entropy
from scipy.optimize import linear_sum_assignment  # hungarian algorithm

from .base import StructuredModel


class Matching(StructuredModel):
    def __init__(self, n_features=None, n_states=None, task="graph"):
        self.n_features = n_features
        self.n_states = n_states
        if task not in ["graph", "ranking"]:
            raise ValueError("Task must be graph or rank.")
        self.task = task
        self._set_size_joint_feature()

    def __repr__(self):
        return ("%s(n_features=%d, n_states=%d)"
                % (type(self).__name__, self.n_features, self.n_states))
        
    def _set_size_joint_feature(self):
        if None not in [self.n_states, self.n_features]:
            if self.task == "ranking":
                self.size_joint_feature = self.n_states * self.n_states * self.n_features
            elif self.task == "graph":
                self.size_joint_feature = self.n_features
            

    def initialize(self, X, Y):
        n_features = X[0].shape[-1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))
        n_states = Y[0].shape[0]
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d classes, got %d"
                             % (self.n_states, n_states))
        self._set_size_joint_feature()

    def output_embedding(self, y):
        # already in embedded representation
        return y

    def barycenters(self, y):
        mu = [np.ones((self.n_states, self.n_states)) / self.n_states, 
                    np.ones((self.n_states, self.n_states)) / self.n_states]
        return mu

    ###########################################################################
    # Features
    ###########################################################################

    def joint_feature(self, x, y):
        if self.task == "ranking":
            result = (np.reshape(x, [-1, 1, 1]) * np.expand_dims(y, 0)).ravel()
        elif self.task == "graph":
            result = (x * np.expand_dims(y, 2)).sum(0).sum(0).ravel()
        assert self.size_joint_feature == result.shape[0]
        return result

    def batch_joint_feature(self, X, Y):
        return sum([self.joint_feature(x, y) for x, y in zip(X, Y)])

    def mean_joint_feature(self, x, q):
        if self.task == "ranking":
            result = (np.reshape(x, [-1, 1, 1]) * np.expand_dims(q, 0)).ravel()
        elif self.task == "graph":
            result = (x * np.expand_dims(q, 2)).sum(0).sum(0).ravel()
        assert self.size_joint_feature == result.shape[0]
        return result

    def batch_mean_joint_feature(self, X, Q, Y_true=None):
        return sum([self.mean_joint_feature(x, q) for x, q in zip(X, Q)])

    ###########################################################################
    # Inference
    ###########################################################################

    def compute_scores(self, x, w):
        if self.task == "ranking":
            n_states = self.n_states
            scores = np.expand_dims(x, 0).dot(np.reshape(w, (self.n_features, self.n_states ** 2)))
        elif self.task == "graph":
            n_states = x.shape[0]
            # pdb.set_trace()
            x = np.transpose(x, (2, 0, 1))
            x = np.reshape(x, (self.n_features, n_states * n_states))
            
            scores = np.dot(np.expand_dims(w, 0), x)
        scores = np.reshape(scores, (n_states,n_states))
        return scores

    def inference(self, x, w, return_energy=False):
        scores = self.compute_scores(x, w)
        n_states = scores.shape[0]
        row_ind, col_ind = linear_sum_assignment(-scores)
        out = np.zeros((n_states, n_states))
        out[row_ind, col_ind] = 1
        if return_energy:
            return out, scores[row_ind, col_ind].sum()
        return out

    def batch_inference(self, X, w):
        return [self.inference(x, w) for x in X]

    ###########################################################################
    # Loss
    ###########################################################################

    def loss(self, y, y_hat):
        n_states = y.shape[0]
        return 1. - (y * y_hat).sum() / n_states

    def batch_loss(self, Y, Y_hat):
        losses = [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]
        return np.array(losses)

    def cond_loss(self, y, q):
        n_states = y.shape[0]
        cond_loss = 1 - y * q / n_states
        return cond_loss[y]

    def batch_cond_loss(self, Y, Q):
        # not implemented for class_weight different
        cond_losses = [self.cond_loss(y, q) for y, q in zip(Y, Q)]
        return np.array(cond_losses)

    def Bayes_risk(self, q):
        n_states = q.shape[0]
        row_ind, col_ind = linear_sum_assignment(-q)
        opt = 1 - q[row_ind, col_ind].sum() / n_states
        return opt

    def batch_Bayes_risk(self, Q):
        bayes_risks = [self.Bayes_risk(q) for q in Q]
        return np.array(bayes_risks)


###############################################################################
# Max Margin Method
###############################################################################


class M3NMatching(Matching):

    def __init__(self, n_features=None, n_classes=None, Loss=None, args=None):
        Matching.__init__(self, n_features=n_features, n_states=n_classes,
                                                            task=args.task)


    def loss_augmented_inference(self, x, y, w):
        scores = self.compute_scores(x, w)
        cost = scores + 1 - y / self.n_states
        row_ind, col_ind = linear_sum_assignment(-cost)
        out = np.zeros((self.n_states, self.n_states))
        out[row_ind, col_ind] = 1
        return out

    def batch_loss_augmented_inference(self, X, Y, w):
        return [self.loss_augmented_inference(x, y, w) for x, y in zip(X, Y)]

    
###############################################################################
# Max-Min Margin Method
###############################################################################


class M4NMatching(Matching):
    
    def __init__(self, n_features=None, n_classes=None, Loss=None, args=None):
        Matching.__init__(self, n_features=n_features,
                                            n_states=n_classes, task=args.task)
        self.inference_type = "P"
        if args.cython:
            self.inference_type = "C"
        self.iter_oracle = args.iter_oracle
        self.iter_oracle_log = args.iter_oracle_log


    def loss_augmented_inference(self, x, mu_init, w,
                                warmstart=True, log=False):
        scores = self.compute_scores(x, w)
        P, Q = mu_init.copy()
        max_iter = self.iter_oracle
        if log: max_iter = self.iter_oracle_log
        if not warmstart:
            P, Q = self.barycenters(None)
        if self.inference_type == "P" or self.inference_type == "C":
            # python implementation of spmp
            L = 2 * np.log(self.n_states)
            eta =  2 * np.log(self.n_states) / (2 * L)
            Q, P, U, V, dg, _ = maxmin_spmp_entropy(P, Q, scores,
                                max_iter, eta, Logs=None)
            err = dg[-1]
        else:
            raise ValueError("inference_type must be C or P.")
        return [P, Q], [U, V], err


    def batch_loss_augmented_inference(self, X, mu_hats, w, log=False):
        return [self.loss_augmented_inference(x, mu_hat, w, log=log)[:2]
                 for x, mu_hat in zip(X, mu_hats)]