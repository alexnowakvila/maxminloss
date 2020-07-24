
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator


class BaseSSVM(BaseEstimator):
    """ABC that implements common functionality."""
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def predict(self, X):
        if hasattr(self.model, 'batch_inference'):
            return self.model.batch_inference(X, self.w)
        return [self.model.inference(x, self.w) for x in X]

    def score(self, X, Y):
        if hasattr(self.model, 'batch_loss'):
            return np.mean(self.model.batch_loss(Y, self.predict(X)))
        return np.mean([self.model.loss(y, y_pred)
                      for y, y_pred in zip(Y, self.predict(X))])
