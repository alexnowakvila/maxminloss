import numpy as np


class StructuredModel(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least joint_feature and inference.
    """
    def __repr__(self):
        return ("%s, size_joint_feature: %d"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionalty of the joint
        features for an instance with labeling (x, y).
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_joint_feature, w.shape))

    def initialize(self, X, Y):
        # set any data-specific parameters in the model
        pass

    def joint_feature(self, x, y):
        raise NotImplementedError()

    def batch_joint_feature(self, X, Y, Y_true=None):
        joint_feature_ = np.zeros(self.size_joint_feature)
        for x, y in zip(X, Y):
            joint_feature_ += self.joint_feature(x, y)
        return joint_feature_

    def inference(self, x, w, relaxed=None):
        raise NotImplementedError()

    def batch_inference(self, X, w, relaxed=None):
        # default implementation of batch inference
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def loss(self, y, y_hat):
        raise NotImplementedError()

    def batch_loss(self, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        raise NotImplementedError()

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        # default implementation of batch loss augmented inference
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]
