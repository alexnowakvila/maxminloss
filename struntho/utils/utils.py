from time import time
import numpy as np
import pdb
import os
import scipy
import scipy.sparse
import pandas as pd
from scipy.linalg import toeplitz
from sklearn import preprocessing


from struntho.datasets.datasets import *
from struntho import models
from struntho.learners import FrankWolfeSSVM, GeneralizedFrankWolfeSSVM, SDCA

def filter_len1(X, y):
    XX, yy = [], []
    for j, x in enumerate(X):
        if x.shape[0] != 1:
            XX.append(x)
            yy.append(y[j])
    return XX, yy

def transform_features_seq(X, M, seed):
    lengths = []
    XX = []
    for x in X:
        XX.extend([x[i] for i in range(x.shape[0])])
        lengths.append(x.shape[0])
    XX = np.array(XX)
    sigma = np.sqrt(np.std(XX, axis=0).sum())
    np.random.seed(seed)
    N = len(XX)
    perm = np.random.permutation(N)
    X1 = np.expand_dims(XX, 1)
    X2 = np.expand_dims(XX[perm][:M], 0)
    X_gaussian = np.exp(-((X1-X2)**2).sum(2)/ (2 * sigma ** 2))
    X_gaussian = preprocessing.scale(X_gaussian)
    # go back to sequence style
    l = 0
    for i, length in enumerate(lengths):
        X[i] = X_gaussian[l: l + length]
        l += length
    return X

def transform_features(X, M, sigma, seed, lst=False):
    if lst: 
        X = np.array(X)
    np.random.seed(seed)
    N = len(X)
    perm = np.random.permutation(N)
    X1 = np.expand_dims(X, 1)
    X2 = np.expand_dims(X[perm][:M], 0)
    X_gaussian = np.exp(-((X1-X2)**2).sum(2)/ (2 * sigma ** 2))
    X_gaussian = preprocessing.scale(X_gaussian)
    if lst: 
        X_gaussian = [X_gaussian[i] for i in range(X_gaussian.shape[0])]
    return X_gaussian


def add_bias(X, n_features):
    for i, x in enumerate(X):
        last = np.ones_like(x)[..., :1]
        x = np.concatenate((x, last), x.ndim - 1)
        X[i] = x
    n_features = n_features + 1
    return X, n_features


def get_model(args):
    model_name = None
    if args.task == "multiclass" or args.task == "ordinal":
        model_name = "MultiClass"
    elif args.task == "sequence":
        model_name = "ChainFactorGraph"
    elif args.task == "ranking" or args.task == "graph":
        model_name = "Matching"

    if args.model == "m3n":
        model_name = "M3N" + model_name
    elif args.model == "m4n":
        model_name = "M4N" + model_name
    elif args.model == "crf":
        model_name = "CRF" + model_name

    return getattr(models, model_name)


def get_learner(args):
    if args.model == "m3n":
        return FrankWolfeSSVM
    elif args.model == "m4n":
        return GeneralizedFrankWolfeSSVM
    elif args.model == "crf":
        return SDCA
    

def get_dataset(base_path, args):
    """ This function is meant to load any dataset from the arguments passed to the main function """
    with open(os.path.join(base_path, "datasets.json"), "r") as file:
        datasets = json.load(file)
    # check whether task and dataset are correct
    if args.task not in datasets.keys():
        raise ValueError("{} is not an existing task".format(args.task))
    elif args.dataset not in datasets[args.task].keys():
        raise ValueError("{} is not an existing dataset".format(args.dataset))
    # load dataset
    dataset_path = os.path.join(base_path, args.task)
    dataset = args.dataset
    with open(os.path.join(dataset_path, "{}/{}.pickle".format(dataset, dataset)), "rb") as file:
        data = pickle.load(file)
    if args.task == "multiclass":
        n_classes = int(data["n_classes"])
        Loss = np.ones((n_classes, n_classes))
        np.fill_diagonal(Loss, 0.0)
        # preprocessing
        data["X"] = preprocessing.scale(data["X"])
    elif args.task == "ordinal":
        n_classes = int(data["n_classes"])
        Loss = toeplitz(np.arange(n_classes)).astype(float)
        # preprocessing
        data["X"] = preprocessing.scale(data["X"])
    elif args.task == "multilabel":
        # preprocessing
        raise NotImplementedError
    elif args.task == "sequence":
        n_classes = int(data["n_classes"])
        Loss = np.ones((n_classes, n_classes))
        np.fill_diagonal(Loss, 0.0)
        # preprocessing
    elif args.task == "ranking":
        n_classes = int(data["n_classes"])
        Loss = None
        # preprocessing
        data["X"] = preprocessing.scale(data["X"])
    elif args.task == "graph":
        n_classes = int(data["n_classes"])
        Loss = None
    return data, Loss

