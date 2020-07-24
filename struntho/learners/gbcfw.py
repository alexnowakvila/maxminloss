import warnings
from time import time
import numpy as np
import pdb;
from sklearn.utils import check_random_state
import pickle

from .ssvm import BaseSSVM


class GeneralizedFrankWolfeSSVM(BaseSSVM):
    def __init__(self, model, logger, X_test=None, Y_test=None, warmstart=True):
        BaseSSVM.__init__(self, model, logger)

        args = logger.args
        self.max_iter = args.epochs
        self.lambd = args.reg
        self.line_search = args.line_search
        self.verbose_samples = args.verbose_samples
        self.check_dual_every = args.check_dual_every
        self.do_averaging = True
        self.sample_method = args.sample_method
        self.random_state = args.random_state
        self.X_test = X_test
        self.Y_test = Y_test
        self.warmstart = warmstart

    def _calc_dual_gap(self, X, Y, nu_hats):
        n_samples = len(X)
        # FIXME don't calculate this again
        joint_feature_gt = self.model.batch_joint_feature(X, Y)
        Q_hat = self.model.batch_loss_augmented_inference(X, nu_hats,
                                                        self.w, log=True)
        Q_hat = np.array([q_hat[0][0] for q_hat in Q_hat])
        djoint_feature = joint_feature_gt - self.model.batch_mean_joint_feature(X, Q_hat)
        ls = np.sum(self.model.batch_Bayes_risk(Q_hat))
        ws = djoint_feature / (self.lambd * n_samples)
        # l = np.sum(self.model.batch_Bayes_risk(mu)) / n_samples
        l_rescaled = self.l * n_samples * self.lambd
        l_rescaled = self.l
        # l_rescaled = l * n_samples * self.lambd
        # dual objective
        dual_objective = -self.lambd / 2 * np.sum(self.w ** 2) + l_rescaled
        # dual gap
        w_diff = self.w - ws
        dual_gap = self.lambd * w_diff.T.dot(self.w) - l_rescaled + ls / n_samples
        # primal objective
        primal_objective = dual_objective + dual_gap
        # primal_val = 0.5 * np.sum(self.w ** 2) + self.lambd * ls - ws.T.dot(self.w)
        self.logger.dual_objective = dual_objective
        self.logger.dual_gap = dual_gap
        self.logger.primal_objective = primal_objective


    def _gfrank_wolfe_bc(self, X, Y):
        n_samples = len(X)
        w = self.w.copy()
        w_mat = np.zeros((n_samples, self.model.size_joint_feature))
        mu = [self.model.output_embedding(y) for y in Y]
        nu_hats = [self.model.barycenters(y) for y in Y]
        l_mat = np.zeros(n_samples)
        l = 0.0
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
                # if iteration > 50:
                #     warmstart = True
                mu_hat, nu_hat, oracle_err = self.model.loss_augmented_inference(x, nu_hats[i], w, warmstart=self.warmstart, log=False)
                oracle_err_avg += oracle_err
                # ws and ls
                delta_joint_feature = self.model.joint_feature(x, y) - self.model.mean_joint_feature(x, mu_hat[0])
                loss = self.model.Bayes_risk(mu_hat[0])
                ws = delta_joint_feature / (self.lambd * n_samples)
                ls = loss / n_samples
                # step-size
                gamma = 2.0 * n_samples / (k + 2.0 * n_samples)
                # update primal variables
                

                w -= w_mat[i]
                w_mat[i] = (1.0 - gamma) * w_mat[i] + gamma * ws
                w += w_mat[i]
                # update warm-start variables
                nu_hats[i] = nu_hat
                # update dual variables
                mu[i] = (1.0 - gamma) * mu[i] + gamma * mu_hat[0]
                # update auxiliary variables
                l_mat_i = self.model.Bayes_risk(mu[i]) / n_samples
                l = l + l_mat_i - l_mat[i] 
                l_mat[i] = l_mat_i
                # copy or do averaging
                if self.do_averaging:
                    rho = 2. / (k + 2.)
                    self.w = (1. - rho) * self.w + rho * w
                    self.l = (1. - rho) * self.l + rho * l
                else:
                    self.w = w
                    self.l = l
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
                    self._calc_dual_gap(X, Y, nu_hats)
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
            self._gfrank_wolfe_bc(X, Y)
        except KeyboardInterrupt:
            pass
        return self
