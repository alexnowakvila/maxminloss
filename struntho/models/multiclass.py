import numpy as np
import cvxopt as cvx
from cvxopt import matrix, solvers
import pdb
import scipy.special as sp
from struntho.utils._testing import assert_allclose
from struntho.inference._maxmin_spmp_multiclass import multiclass_oracle_c
from struntho.inference.maxmin_spmp_multiclass import maxmin_multiclass_cvxopt
from struntho.inference.maxmin_spmp_multiclass import maxmin_spmp_multiclass_p
from scipy.special import kl_div, softmax
from scipy.stats import entropy as entr

from .base import StructuredModel


###############################################################################
# Multiclass Base Class
###############################################################################


class MultiClass(StructuredModel):

    def __init__(self, n_features=None, n_classes=None, Loss=None):
        self.n_features = n_features
        self.n_states = n_classes
        self._set_size_joint_feature()
        if Loss is None:
            # set to 0-1 loss
            self.Loss = (np.dot(np.expand_dims(np.ones, 1),
                            np.ones((1, np.ones))))
            np.fill_diagonal(self.Loss, 0.0)
        else:
            self.Loss = Loss

    def __repr__(self):
        return ("%s(n_features=%d, n_classes=%d)"
                % (type(self).__name__, self.n_features, self.n_states))
        
    def _set_size_joint_feature(self):
        if None not in [self.n_states, self.n_features]:
            self.size_joint_feature = self.n_states * self.n_features

    def initialize(self, X, Y):
        n_features = X[0].shape[0]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_classes = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_classes
        elif self.n_states != n_classes:
            raise ValueError("Expected %d classes, got %d"
                             % (self.n_states, n_classes))
        self._set_size_joint_feature()

    def output_embedding(self, y):
        mu = np.zeros(self.n_states)
        mu[y] = 1
        return mu

    def barycenters(self, y):
        mu = [np.ones(self.n_states) / self.n_states,
                    np.ones(self.n_states) / self.n_states]
        return mu

    ###########################################################################
    # Features
    ###########################################################################


    def joint_feature(self, x, y):
        result = np.zeros((self.n_states, self.n_features))
        result[y, :] = x
        return result.ravel()

    def batch_joint_feature(self, X, Y):
        result = np.zeros(self.size_joint_feature)
        for i in range(len(X)):
            result += self.joint_feature(X[i], Y[i])
        return result

    def mean_joint_feature(self, x, q):
        result = np.ones((self.n_states, self.n_features))
        result = result * x.reshape(1, -1)
        result = result * q.reshape(-1, 1)
        return result.ravel()

    def batch_mean_joint_feature(self, X, Q):
        result = np.zeros(self.size_joint_feature)
        for i in range(len(X)):
            result += self.mean_joint_feature(X[i], Q[i])
        return result

    def score_function(self, x, w, y):
        joint_feature = self.joint_feature(x, y)
        return (w * joint_feature).sum()

    ###########################################################################
    # Inference
    ###########################################################################

    def compute_scores(self, x, w):
        return np.dot(w.reshape(self.n_states, -1), x)

    def inference(self, x, w, return_energy=False):
        scores = self.compute_scores(x, w)
        if return_energy:
            return np.argmax(scores), np.max(scores)
        return np.argmax(scores)

    def batch_inference(self, X, w):
        scores = np.dot(np.array(X), w.reshape(self.n_states, -1).T)
        return np.argmax(scores, axis=1)

    ###########################################################################
    # Loss
    ###########################################################################

    def loss(self, y, y_hat):
        return self.Loss[y, y_hat]

    def batch_loss(self, Y, Y_hat):
        losses = [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]
        return np.array(losses)

    def cond_loss(self, y, q):
        cond_loss = np.dot(self.Loss, np.expand_dims(q, 1))
        return cond_loss[y]

    def batch_cond_loss(self, Y, Q):
        cond_losses = [self.cond_loss(y, q) for y, q in zip(Y, Q)]
        return np.array(cond_losses)

    def Bayes_risk(self, q):
        cond_loss = np.dot(self.Loss, np.expand_dims(q, 1))
        opt = np.min(cond_loss)
        return opt

    def batch_Bayes_risk(self, Q):
        bayes_risks = [self.Bayes_risk(q) for q in Q]
        return np.array(bayes_risks)


###############################################################################
# Multinomial Logistic Regression
###############################################################################


class CRFMultiClass(MultiClass):
    
    def __init__(self, n_features=None, n_classes=None, Loss=None, args=None):

        MultiClass.__init__(self, n_features=n_features,
                                n_classes=n_classes,
                                Loss=Loss)
                                
    def marginal_inference(self, x, w):
        scores = self.compute_scores(x, w)
        return softmax(scores)

    def batch_marginal_inference(self, X, w):
        return [self.marginal_inference(x, w) for x in X]

    def inference(self, x, w, return_energy=False):
        q = self.marginal_inference(x, w)
        cond_loss = np.dot(self.Loss, np.expand_dims(q, 1)).flatten()
        if return_energy:
            return np.argmin(cond_loss), np.min(cond_loss)
        return np.argmin(cond_loss)

    def entropy(self, mu):
        return entr(mu)
    
    def batch_entropy(self, MU):
        return [self.entropy(mu) for mu in MU]

    def kl(self, mu1, mu2):
        return kl_div(mu1, mu2)
    
    def batch_kl(self, MU1, MU2):
        return [self.kl(mu1, mu2) for mu1, mu2 in zip(MU1, MU2)]

    def entropy_derivative(self, mu):
        return - np.log(mu) -1

    def partition_function(self, x, w):
        scores = self.compute_scores(x, w)
        return np.log(np.exp(scores).sum() + 1e-6)

    def batch_partition_function(self, X, w):
        return [self.partition_function(x, w) for x in X]


###############################################################################
# Crammer-Singer SVM
###############################################################################


class M3NMultiClass(MultiClass):
    
    def __init__(self, n_features=None, n_classes=None, Loss=None, args=None):

        MultiClass.__init__(self, n_features=n_features,
                                n_classes=n_classes,
                                Loss=Loss)

    def loss_augmented_inference(self, x, y, w):
        scores = self.compute_scores(x, w)
        other_classes = np.arange(self.n_states) != y
        add = self.Loss[np.arange(self.n_states), y]
        scores += add
        return np.argmax(scores)

    def batch_loss_augmented_inference(self, X, Y, w):
        return [self.loss_augmented_inference(x, y, w) for x, y in zip(X, Y)]


###############################################################################
# Max-Min Multiclass
###############################################################################


class M4NMultiClass(MultiClass):
    
    def __init__(self, n_features=None, n_classes=None, Loss=None, args=None):

        MultiClass.__init__(self, n_features=n_features,
                                n_classes=n_classes,
                                Loss=Loss)
        self.inference_type = "P"
        if args.cython:
            self.inference_type = "C"
        self.iter_oracle = args.iter_oracle
        self.iter_oracle_log = args.iter_oracle_log
        

    def loss_augmented_inference(self, x, mu_init, w,
                                warmstart=True, log=False):
        scores = self.compute_scores(x, w)
        nu, p = mu_init.copy()
        max_iter = self.iter_oracle
        if not warmstart:
            nu, p = self.barycenters(None)
        if log: max_iter = self.iter_oracle_log
        if self.inference_type == "E":
            # exact using cvxopt library
            mu, en, err , _ = maxmin_multiclass_cvxopt(scores, self.Loss)
            q = None
        elif self.inference_type == "P":
            # python implementation of spmp
            L = np.max(self.Loss) * np.log(self.n_states)
            eta =  np.log(self.n_states) / (2 * L)
            mu, q, nu, p, dg, _ = maxmin_spmp_multiclass_p(nu, p, scores,
                                self.Loss, max_iter, eta, Logs=None)
            err = dg[-1]
        elif self.inference_type == "C":
            # cython implementation of spmp
            L = np.max(self.Loss) * np.log(self.n_states)
            eta =  np.log(self.n_states) / (2 * L)
            mu, q, nu, p, dg = multiclass_oracle_c(nu.copy(), p.copy(), scores,
                                self.Loss, max_iter, eta)
            mu, q = np.array(mu), np.array(q) 
            nu, p = np.array(nu), np.array(p) 
            dg = np.array(dg)
            err = dg[-1]
        else:
            raise ValueError("inference_type must be C, P or E.")
        return [mu, q], [nu, p], err

    def batch_loss_augmented_inference(self, X, MU, w, log=False):
        return [self.loss_augmented_inference(x, mu, w, log=log)[:2]
                                for x, mu in zip(X, MU)]
        
        
