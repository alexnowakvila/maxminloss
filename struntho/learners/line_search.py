import numpy as np
import pdb
from scipy.optimize import minimize_scalar


class LineSearch:
    def __init__(self, model, w, v_i, mu, delta_i, lambd,
                                n_samples, subprecision=0.001):
        self.model = model
        self.w = w
        self.v_i = v_i
        self.mu = mu
        self.delta_i = delta_i
        self.lambd = lambd
        self.n_samples = n_samples
        self.subprecision = subprecision

    def evaluator(self, step_size, return_f=False, 
                                return_df=False, return_newton=False):
        ans = []
        if return_f:
            ans.append(self._function(step_size))
        if return_df:
            df = self._derivative(step_size)
            ans.append(df)
        return tuple(ans) if len(ans) > 1 else ans[0]

    def _function(self, step_size):
        w_upd = self.w + step_size * self.v_i
        mu_upd = self.mu + step_size * self.delta_i
        coef = self.lambd * self.n_samples
        # pdb.set_trace()
        return self.model.entropy(mu_upd) - coef / 2 * np.sum(w_upd ** 2)

    def _derivative(self, step_size):
        w_upd = self.w + step_size * self.v_i
        mu_upd = self.mu + step_size * self.delta_i
        coef = self.lambd * self.n_samples
        der1 = (self.model.entropy_derivative(mu_upd) * self.delta_i).sum()
        der2 = - coef * (w_upd * self.v_i).sum()
        return der1 + der2

    def run(self):
        return self.run_scipy()

    def run_scipy(self):
        result = minimize_scalar(lambda x: -1*self.evaluator(x, return_f=True),
                                 bounds=(0, 1), method='bounded',
                                 options={'xatol': self.subprecision})
        optimal_step_size = result.x
        return optimal_step_size