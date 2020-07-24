import numpy as np
import pdb

from .base import StructuredModel

from struntho.inference.maxmin_spmp_sequence import maxmin_spmp_sequence_p, maxmin_spmp_sequence_p2
from struntho.inference.sum_product_chain import sum_product_p
from struntho.inference._maxmin_spmp_sequence import maxmin_spmp_sequence_c, maxmin_spmp_sequence_c2
from struntho.inference._sum_product_chain import viterbi, sum_product_c, log_partition_c
from scipy.special import kl_div
from scipy.stats import entropy as entr

import numpy as np

###############################################################################
# PairWise Factor Graph
###############################################################################

class PWFactorGraph(StructuredModel):
    """Pairwise Factor Graph"""
    def __init__(self, n_states=None, n_features=None, Loss=None):
        self.n_states = n_states
        # if inference_method is None:
        #     # get first in list that is installed
        #     inference_method = get_installed(['ad3', 'max-product', 'lp'])[0]
        self.n_features = n_features
        self._set_size_joint_feature()
        self.Loss = Loss

    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s)"
                % (type(self).__name__, self.n_states,
                   self.inference_method))


    def _set_size_joint_feature(self):
        # try to set the size of joint_feature if possible
        if self.n_features is not None and self.n_states is not None:
            self.size_joint_feature = (self.n_states * self.n_features +
                                           self.n_states * self.n_states)

    def _get_unary_potentials(self, x, w):
        self._check_size_w(w)
        self._check_size_x(x)
        features = self._get_features(x)
        unary_params = w[:self.n_states * self.n_features].reshape(
            self.n_states, self.n_features)
        return np.dot(features, unary_params.T)

    def _get_pairwise_potentials(self, x, w):
        self._check_size_w(w)
        self._check_size_x(x)
        pw = w[self.n_states * self.n_features:]
        return pw.reshape(self.n_states, self.n_states)

    def _get_edges(self, x):
        raise NotImplementedError

    def _get_features(self, x):
        raise NotImplementedError

    def initialize(self, X, Y):
        # Works for both GridCRF and GraphCRF, but not ChainCRF.
        # funny that ^^
        # pdb.set_trace()
        # n_features = 
        # if self.n_features is None:
        #     self.n_features = n_features
        # elif self.n_features != n_features:
        #     raise ValueError("Expected %d features, got %d"
        #                      % (self.n_features, n_features))

        # n_states = len(np.unique(np.hstack([y.ravel() for y in Y])))
        # if self.n_states is None:
        #     self.n_states = n_states
        # elif self.n_states != n_states:
        #     raise ValueError("Expected %d states, got %d"
        #                      % (self.n_states, n_states))
        self._set_size_joint_feature()


    def _check_size_x(self, x):
        features = self._get_features(x)
        if features.shape[1] != self.n_features:
            raise ValueError("Unary evidence should have %d feature per node,"
                             " got %s instead."
                             % (self.n_features, features.shape[1]))

    def output_embedding(self, y):
        length = y.shape[0]
        # set unaries embeddings
        node_embeddings = np.zeros((length, self.n_states))
        node_embeddings[np.arange(length), y] = 1
        # set pairwise embeddings
        edge_embeddings = np.expand_dims(node_embeddings, 1)[:-1] * np.expand_dims(node_embeddings, 2)[1:]
        # flatten
        mu = np.hstack([node_embeddings.ravel(), edge_embeddings.ravel()])
        return mu

    def barycenters(self, y):
        length = y.shape[0]
        mu1 = np.ones(length * self.n_states) / self.n_states
        mu2 = np.ones((length-1) * self.n_states * self.n_states) / (self.n_states * self.n_states)
        mu3 = np.hstack([mu1, mu2])
        q = np.ones(length * self.n_states) / self.n_states
        mu = [mu3, q]
        return mu

    def extract_marginals(self, mu, nodes_only=False):
        try:
            length = (mu.shape[0] + self.n_states**2) // (self.n_states**2+self.n_states)
        except AttributeError:
            raise ValueError("Dimensions are not correct.")
        mu_length_comp = (length - 1) * self.n_states ** 2 + length * self.n_states
        assert mu_length_comp  == mu.shape[0]
        mu_nodes = mu[:length * self.n_states].reshape(length, self.n_states)
        if nodes_only:
            return mu_nodes
        else:
            mu_edges = mu[length * self.n_states:].reshape(length-1, self.n_states, self.n_states)
        return mu_nodes, mu_edges

    ###########################################################################
    # Features
    ###########################################################################

    def joint_feature(self, x, y):
        # THIS CAN BE CHANGED IN THE FORM OF MEAN JOINT FEATURE
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]

        y = y.reshape(n_nodes)
        gx = np.ogrid[:n_nodes]

        #make one hot encoding
        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
        gx = np.ogrid[:n_nodes]
        unary_marginals[gx, y] = 1

        ##accumulated pairwise
        pw = np.dot(unary_marginals[edges[:, 0]].T,
                    unary_marginals[edges[:, 1]])

        unaries_acc = np.dot(unary_marginals.T, features)
        pw = pw.ravel()
        joint_feature_vector = np.hstack([unaries_acc.ravel(), pw])
        return joint_feature_vector

    def batch_joint_feature(self, X, Y):
        result = np.zeros(self.size_joint_feature)
        for i in range(len(X)):
                result += self.joint_feature(X[i], Y[i])
        return result

    def mean_joint_feature(self, x, mu):
        mu_nodes, mu_edges = self.extract_marginals(mu)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]
        unaries_acc = np.dot(mu_nodes.T, features)
        pw = mu_edges.sum(0) # from (length-1, n_classes**2) to (n_classes**2)
        mean_joint_feature_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return mean_joint_feature_vector

    def batch_mean_joint_feature(self, X, MU):
        result = np.zeros(self.size_joint_feature)
        for i in range(len(X)):
                result += self.mean_joint_feature(X[i], MU[i])
        return result

    def score_function(self, x, w, y):
        joint_feature = self.joint_feature(x, y)
        return (w * joint_feature).sum()

    ###########################################################################
    # Inference
    ###########################################################################

    def inference(self, x, w, return_energy=False):
        raise NotImplementedError

    def batch_inference(self, X, w):
        return [self.inference(x, w) for x in X]

    ###########################################################################
    # Loss
    ###########################################################################

    def loss(self, y, y_hat):
        M = y.shape[0]
        L = sum([self.Loss[y[m], y_hat[m]] for m in range(M)])
        return L / M

    def batch_loss(self, Y, Y_hat):
        losses = [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]
        return np.array(losses)

    def cond_loss(self, y, mu):
        mu_nodes = self.extract_marginals(mu, nodes_only=True)
        return np.dot(mu_nodes, self.Loss.T)[y]

    def batch_cond_loss(self, Y, MU):
        cond_losses = [self.cond_loss(y, mu) for y, mu in zip(Y, MU)]
        return np.array(cond_losses)

    def Bayes_risk(self, mu):
        mu_nodes = self.extract_marginals(mu, nodes_only=True)
        cond_loss = np.dot(mu_nodes, self.Loss.T) # shape length * n_classes
        bayes_risk = cond_loss.min(1).sum()
        return bayes_risk

    def batch_Bayes_risk(self, MU):
        bayes_risks = [self.Bayes_risk(mu) for mu in MU]
        return np.array(bayes_risks)


###############################################################################
# Chain Factor Graph
###############################################################################


class ChainFactorGraph(PWFactorGraph):
    """Pairwise Factor Graph"""
    def __init__(self, n_states=None, n_features=None, Loss=None):
        PWFactorGraph.__init__(self, n_states=n_states,
                                    n_features=n_features,
                                    Loss=Loss)
        
    def make_chain_edges(self, x):
        # this can be optimized sooooo much!
        inds = np.arange(x.shape[0])
        edges = np.concatenate([inds[:-1, np.newaxis],
                            inds[1:, np.newaxis]], axis=1)
        return edges

    def _get_edges(self, x):
        return self.make_chain_edges(x)

    def _get_features(self, x):
        return x

    def compute_scores(self, x, w):
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        return unary_potentials, pairwise_potentials

    def inference(self, x, w, return_energy=False):
        self._check_size_w(w)
        unary_potentials, pairwise_potentials = self.compute_scores(x, w)
        edges = self._get_edges(x)
        n_nodes = unary_potentials.shape[0]
        ymax = np.zeros(n_nodes, dtype=np.int32)
        viterbi(unary_potentials, pairwise_potentials, ymax)
        return ymax


###############################################################################
# CRF Chain Factor Graph
###############################################################################


class CRFChainFactorGraph(ChainFactorGraph):
    def __init__(self, n_classes=None, n_features=None, Loss=None, args=None):
        ChainFactorGraph.__init__(self, n_states=n_classes,
                                        n_features=n_features,
                                        Loss=Loss)
        self.inference_type = "P"
        if args.cython:
            self.inference_type = "C"
        
    def marginal_inference(self, x, w):
        # compute vector of scores
        self._check_size_w(w)
        unary_potentials, pairwise_potentials = self.compute_scores(x, w)
        edges = self._get_edges(x)
        length = unary_potentials.shape[0]
        # pairwise_potentials = np.tile(pairwise_potentials, (length, 1, 1))
        pairwise_potentials = np.ones((length - 1, 1, 1)) * np.expand_dims(pairwise_potentials, 0)
        # compute marginal inference
        n_p = np.empty([length, self.n_states])
        e_p = np.empty([length - 1, self.n_states, self.n_states])
        if self.inference_type == "P":
            sum_product_p(unary_potentials, pairwise_potentials, n_p, e_p)
        elif self.inference_type == "C":
            sum_product_c(unary_potentials, pairwise_potentials, n_p, e_p)
        else:
            raise ValueError("inference_type must be C or P.")
        n_p, n_e = np.array(np.exp(n_p)).ravel(), np.array(np.exp(e_p)).ravel()
        mu = np.hstack([n_p, n_e])
        return mu

    def batch_marginal_inference(self, X, w):
        return [self.marginal_inference(x, w) for x in X]

    def entropy(self, mu):
        mu_nodes, mu_edges = self.extract_marginals(mu)
        entr_nodes = [entr(mu_n) for mu_n in mu_nodes]
        entr_edges = [entr(mu_e.ravel()) for mu_e in mu_edges]
        return sum(entr_edges) - sum(entr_nodes) 
    
    def batch_entropy(self, MU):
        return [self.entropy(mu) for mu in MU]

    def kl(self, mu1, mu2):
        mu_nodes1, mu_edges1 = self.extract_marginals(mu1)
        mu_nodes2, mu_edges2 = self.extract_marginals(mu2)
        length = mu_nodes1.shape[0]
        kl_nodes = []
        # for mu1, mu2 in zip(mu_nodes1, mu_nodes2):
        #     pdb.set_trace()
        #     kl_nodes.append(kl_div(mu1, mu2).sum())
        kl_nodes = [kl_div(mu1, mu2).sum() for mu1, mu2 in zip(mu_nodes1, mu_nodes2)]
        kl_edges = [kl_div(mu1.ravel(), mu2.ravel()).sum() 
                                        for mu1, mu2 in 
                                        zip(mu_edges1, mu_edges2)]
        return sum(kl_edges) - sum(kl_nodes) 
    
    def batch_kl(self, MU1, MU2):
        return [self.kl(mu1, mu2) for mu1, mu2 in zip(MU1, MU2)]

    def partition_function(self, x, w):
        self._check_size_w(w)
        unary_potentials, pairwise_potentials = self.compute_scores(x, w)
        length = unary_potentials.shape[0]
        pairwise_potentials = np.tile(pairwise_potentials, (length - 1, 1, 1))
        return log_partition_c(unary_potentials, pairwise_potentials)

    def batch_partition_function(self, X, w):
        return [self.partition_function(x, w) for x in X]



###############################################################################
# Max Margin Chain Factor Graph
###############################################################################


class M3NChainFactorGraph(ChainFactorGraph):
    def __init__(self, n_classes=None, n_features=None, Loss=None, args=None):
        ChainFactorGraph.__init__(self, n_states=n_classes,
                                        n_features=n_features,
                                        Loss=Loss)
        
    def loss_augmented_inference(self, x, y, w):
        self._check_size_w(w)
        unary_potentials, pairwise_potentials = self.compute_scores(x, w)
        edges = self._get_edges(x)
        n_nodes = unary_potentials.shape[0]
        for j in range(n_nodes):
            unary_potentials[j] += self.Loss[y[j]] / n_nodes
        ymax = np.zeros(n_nodes, dtype=np.int32)
        viterbi(unary_potentials, pairwise_potentials, ymax)
        return ymax

    def batch_loss_augmented_inference(self, X, Y, w):
        return [self.loss_augmented_inference(x, y, w) for x, y in zip(X, Y)]



###############################################################################
# Max-Min Margin Chain Factor Graph
###############################################################################


class M4NChainFactorGraph(ChainFactorGraph):
    def __init__(self, n_classes=None, n_features=None, Loss=None, args=None):
        ChainFactorGraph.__init__(self, n_states=n_classes,
                                        n_features=n_features,
                                        Loss=Loss)
        self.inference_type = "P"
        if args.cython:
            self.inference_type = "C"
        self.iter_oracle = args.iter_oracle
        self.iter_oracle_log = args.iter_oracle_log

    def loss_augmented_inference(self, x, mu_hat, w,
                                 return_energy=False,
                                 warmstart=True,
                                 log=False):
        self._check_size_w(w)
        unary_potentials, pairwise_potentials = self.compute_scores(x, w)
        edges = self._get_edges(x)

        # length = y.shape[0]
        max_iter = self.iter_oracle
        if log: max_iter = self.iter_oracle_log
        if warmstart:
            init = [mu_hat[0].copy(), mu_hat[1].copy()]
        else:
            init = self.barycenters(unary_potentials)
        nu_nodes, nu_edges = self.extract_marginals(init[0].copy())
        length = unary_potentials.shape[0]
        p = np.reshape(init[1], (length, self.n_states))
        eta = 1 / (2 * np.max(self.Loss))
        if self.inference_type == "P":
            out1, out2, dual_gap = maxmin_spmp_sequence_p(nu_nodes,
                                            nu_edges,
                                            p,
                                            unary_potentials,
                                            pairwise_potentials,
                                            self.Loss,
                                            max_iter,
                                            eta,
                                            sum_product_cython=True)
        elif self.inference_type == "C":
            out1, out2, dual_gap = maxmin_spmp_sequence_c(nu_nodes,
                                            nu_edges,
                                            p,
                                            unary_potentials,
                                            pairwise_potentials,
                                            self.Loss,
                                            max_iter,
                                            eta)
            out1 = [[np.array(out1[0][0]), np.array(out1[0][1])],
                        np.array(out1[1])]
            out2 = [[np.array(out2[0][0]), np.array(out2[0][1])],
                        np.array(out2[1])]
        else:
            raise ValueError("inference_type must be C or P.")
        dual_gap = np.array(dual_gap)
        err = dual_gap[-1]
        mu = ( np.hstack([np.array(out1[0][0]).ravel(),
                        np.array(out1[0][1]).ravel()]) )
        q = np.array(out1[1]).ravel()
        nu = ( np.hstack([np.array(out2[0][0]).ravel(),
                        np.array(out2[0][1]).ravel()]) )
        p = np.array(out2[1]).ravel()
        return [mu, q], [nu, p], err

    def batch_loss_augmented_inference(self, X, MU, w, log=False):
        return [self.loss_augmented_inference(x, mu, w, log=log)[:2]
                                    for x, mu in zip(X, MU)]
    

    

