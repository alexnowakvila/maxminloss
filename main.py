from time import time
import numpy as np
import pdb
import os
import scipy.sparse
import pandas as pd

from sklearn import datasets
# from sklearn.datasets import fetch_mldata
# from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.linalg import toeplitz

from struntho.utils.logger import Logger
from struntho.utils.arguments import get_args, init_logdir, get_info
from struntho.utils.utils import get_model, get_learner, get_dataset, add_bias
from struntho.utils.utils import transform_features, transform_features_seq, filter_len1

args = get_args()
np.random.seed(args.random_state)

# load model
Model = get_model(args)

# load learner
Learner = get_learner(args)

# load datasets
dataset, Loss = get_dataset("struntho/datasets", args)

if args.n_samples is not None:
    if dataset["n_samples"] > args.n_samples:
        dataset["n_samples"] = args.n_samples

if args.kernel:
    print("Constructing Nystrom features...")
    M = 200
    M = np.clip(M, 0, dataset["n_samples"])
    if args.task == "sequence":
        dataset["X"] = transform_features_seq(dataset["X"], M, 0)
    else:
        sigma = np.sqrt(np.std(np.array(dataset["X"]), axis=0).sum())
        # sigma = np.sqrt(dataset["n_features"])
        dataset["X"] = transform_features(dataset["X"], M, sigma, 0)
    dataset["n_features"] = M

if args.task == "sequence":
    dataset["X"], dataset["y"] = filter_len1(dataset["X"], dataset["y"])

if "X" in dataset.keys():
    X, y = dataset["X"], dataset["y"]
    perm = np.random.permutation(len(X))
    X, y = [X[i] for i in perm], [y[i] for i in perm]
    X, y = X[:int(dataset["n_samples"])], y[:int(dataset["n_samples"])]
    if args.add_bias:
        X, dataset["n_features"] = add_bias(X, dataset["n_features"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.random_state)
    n_train, n_test = len(X_train), len(X_test)
elif "X_train" in dataset.keys():
    X_test, X_train = dataset['X_test'], dataset['X_train']
    y_test, y_train = dataset['y_test'], dataset['y_train']
    # to lists
    X_train = [X_train[i] for i in range(X_train.shape[0])]
    X_test = [X_test[i] for i in range(X_test.shape[0])]
    y_train = [y_train[i] for i in range(y_train.shape[0])]
    y_test = [y_test[i] for i in range(y_test.shape[0])]

    if args.add_bias:
        n_features = dataset["n_features"]
        X_train, dataset["n_features"] = add_bias(X_train,
                                                    dataset["n_features"])
        X_test, n_features = add_bias(X_test, n_features)
    n_train, n_test = len(X_train), len(X_test)

if args.reg < 0: args.reg = 1 / n_train

infostring = get_info(args, dataset["n_features"], dataset["n_classes"],
                                            n_train, n_test)
init_logdir(args, infostring)
print(infostring)

# initialize logger
logger = Logger(args)
# initialize model
model = Model(n_features=dataset["n_features"],
            n_classes=dataset["n_classes"], Loss=Loss, args=args)
# intitialize learner
learner = Learner(model, logger, X_test=X_test, Y_test=y_test)
learner.fit(X_train, y_train)