import warnings
from time import time
import numpy as np
import pdb;
from sklearn.utils import check_random_state
import pickle
from .line_search import LineSearch
from .ssvm import BaseSSVM


class SDCA(BaseSSVM):
    def __init__(self, model, logger, X_test=None, Y_test=None):
        BaseSSVM.__init__(self, model, logger)

        args = logger.args
        self.max_iter = args.epochs
        self.lambd = args.reg
        self.line_search = args.line_search
        self.verbose_samples = args.verbose_samples
        self.check_dual_every = args.check_dual_every
        self.do_averaging = False
        self.sample_method = args.sample_method
        self.random_state = args.random_state
        self.X_test = X_test
        self.Y_test = Y_test
        self.line_search = args.line_search

    def _calc_dual_gap(self, X, Y, mu):
        n_samples = len(X)
        joint_feature_gt = self.model.batch_joint_feature(X, Y)
        djoint_feature = joint_feature_gt - self.model.batch_mean_joint_feature(X, mu)
        w_hat = djoint_feature / (self.lambd * n_samples)
        mu_hats = self.model.batch_marginal_inference(X, w_hat)
        dual_gap = np.mean(self.model.batch_kl(mu, mu_hats))
        # dual_gap = np.mean(self.model.batch_kl(mu_hats, mu))
        dual_objective = (-self.lambd / 2 * np.sum(self.w ** 2) 
                            + np.mean(self.model.batch_entropy(mu)))
        primal_objective = dual_objective + dual_gap
        # primal_objective = (sum(self.model.batch_partition_function(X, self.w))
        # -sum([self.model.score_function(X[i], self.w, Y[i]) for i in range(n_samples)])) / n_samples + self.lambd / 2 * np.sum(self.w ** 2)
        # dual_gap = primal_objective - dual_objective
        self.logger.dual_objective = dual_objective
        self.logger.dual_gap = dual_gap
        self.logger.primal_objective = primal_objective


    def _sdca_bc(self, X, Y):
        n_samples = len(X)
        # intialize mu
        eps = 0.5
        mu = [eps * self.model.barycenters(y)[0] + (1 - eps) * self.model.output_embedding(y) for y in Y]
        joint_feature_gt = self.model.batch_joint_feature(X, Y)
        djoint_feature = joint_feature_gt - self.model.batch_mean_joint_feature(X, mu)
        # initialize w
        w = djoint_feature / (self.lambd * n_samples)
        k = 0

        rng = check_random_state(self.random_state)
        iterations = []
        for iteration in range(self.max_iter):
            perm = np.arange(n_samples)
            if self.sample_method == 'perm':
                rng.shuffle(perm)
            elif self.sample_method == 'rnd':
                perm = rng.randint(low=0, high=n_samples, size=n_samples)
            # loop over dataset
            oracle_err_avg = 0
            for j in range(n_samples):
                i = perm[j]
                x, y = X[i], Y[i]
                # loss augmented inference
                mu_hat = self.model.marginal_inference(x, w)
                # pdb.set_trace()
                # ascent direction
                delta_i = mu_hat - mu[i]
                # primal direction
                # delta_joint_feature = self.model.joint_feature(x, y) - self.model.mean_joint_feature(x, delta_i)
                # v_i = delta_joint_feature / (self.lambd * n_samples)
                v_i = -self.model.mean_joint_feature(x, delta_i) / (self.lambd * n_samples)
                # line search / step-size
                ls = LineSearch(self.model, w, v_i, mu[i], delta_i,
                                            self.lambd, n_samples)
                if self.line_search:
                    gamma = ls.run()
                else:
                    gamma = 2.0 * n_samples / (k + 2.0 * n_samples)
                # update dual variables
                # pdb.set_trace()
                mu[i] = mu[i] + gamma * delta_i
                # update primal variables
                w = w + gamma * v_i
                # copy (no averaging)
                # copy or do averaging
                if self.do_averaging:
                    rho = 2. / (k + 2.)
                    self.w = (1. - rho) * self.w + rho * w
                else:
                    self.w = w
                k += 1
                # log
                if j % self.verbose_samples == 0 and self.verbose_samples >= 0:
                    # print("Error oracles %f", oracle_err_avg / (j + 1))
                    self.logger.train_error_batch = self.score(X, Y)
                    self.logger.test_error_batch = self.score(self.X_test,
                                                                self.Y_test)
                    self.logger.iteration_batch = k
                    self.logger.append_results_batch()
                    print("iteration {} | sample {} | train loss {:.4f} | test loss {:.4f}".format(iteration,
                                        j, 
                                        self.logger.train_error_batch,
                                        self.logger.test_error_batch))
            # log
            self.logger.oracles.append(oracle_err_avg / n_samples)
            if (self.check_dual_every != 0) and (iteration % self.check_dual_every == 0):
                if self.check_dual_every > 0:
                    self._calc_dual_gap(X, Y, mu)
                self.logger.train_error = self.score(X, Y)
                self.logger.test_error = self.score(self.X_test, self.Y_test)
                self.logger.iteration = iteration
                self.logger.append_results()
                self.logger.save_results()
                print(self.logger)

    def fit(self, X, Y, constraints=None, initialize=True):
        if initialize:
            self.model.initialize(X, Y)
        self.w = getattr(self, "w", np.zeros(self.model.size_joint_feature))
        self.l = getattr(self, "l", 0)
        self.logger.oracles = []
        try:
            self._sdca_bc(X, Y)
        except KeyboardInterrupt:
            pass
        return self
