import argparse
import os
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='struntho')

    ###########################################################################
    # General settings
    ###########################################################################

    parser.add_argument('--task', type=str, choices=["multiclass", "multilabel", "ordinal", "sequence", "ranking", "graph"], default='multiclass',
                        help='which dataset to use')
    parser.add_argument('--dataset', type=str, default='iris',
                        help='which dataset to use')
    parser.add_argument('--model', type=str, choices=["m3n", "m4n", "crf"], default='m3n', help='model to use')
    parser.add_argument('--kernel', action='store_true')
    parser.add_argument('--M', type=int, default=200,
                        help='Nystrom points')
    parser.add_argument('--add_bias', action='store_true')

    ###########################################################################
    # General learning arguments
    ###########################################################################

    parser.add_argument('--n_samples', type=int, default=None,
                        help='set to None if you want the full data set.')
    parser.add_argument('--reg', type=float, default=-1,
                        help='value of the l2 reg parameter. '
                             'if negative, will be set to 1/n.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum number of pass over the trainset duality gaps used in the '
                             'non-uniform sampling and to get a convergence criterion.')
    parser.add_argument('--splits', type=int, default=15,
                        help='number of splits in crossvalidation')
    parser.add_argument('--line_search', action='store_true')
    parser.add_argument('--sample_method', type=str, choices=['perm', 'rnd', 'seq'], default="perm")
    parser.add_argument('--random_state', type=int, default=0,
                        help='random state for sampling in learning')
    parser.add_argument('--cython', action='store_true')
    parser.add_argument('--save', action='store_true')
    # log frequency
    parser.add_argument('--verbose_samples', type=int, default=-1,
                        help='frequency to which compute loss')
    parser.add_argument('--check_dual_every', type=int, default=10,
                        help='frequency to which compute objective values')

    ###########################################################################
    # GBCFW learning arguments
    ###########################################################################

    parser.add_argument('--warmstart', type=int, default=1,
                        help='whether to warmstart or not')
    parser.add_argument('--iter_oracle', type=int, default=10,
                        help='spmp iterations in the oracle')
    parser.add_argument('--iter_oracle_log', type=int, default=10,
                        help='spmp iterations in the oracle to compute the dual gap')
    

    args = parser.parse_args()

    args.time_stamp = time.strftime("%Y%m%d_%H%M%S")
    return args


def get_logdir(args, cv=None, iter_oracle=None, warmstart=1):
    path = "logs/{}_{}_{}/{}_{:.10f}".format(
        args.task, args.model, args.dataset,
        args.time_stamp,
        args.reg)
    if cv is not None:
        path = "logs/{}_{}_{}/{}_{:.10f}_cv{}".format(
            args.task, args.model, args.dataset,
            args.time_stamp,
            args.reg,
            cv)
    if iter_oracle is not None:
        path = "logs/{}_{}_{}/{}_{:.10f}_{}_iter_oracle{:03d}".format(
            args.task, args.model, args.dataset,
            args.time_stamp,
            args.reg,
            warmstart,
            iter_oracle)
    return path


def get_info(args, n_features, n_states, n_train, n_test):
    info = "\n"
    info += "|".join(["-"] * 80) + "\n"
    info += "task: {} \n dataset: {} \n model: {} \n reg: {:.4f} \n n_features: {} \n n_states: {} \n n_train: {} \n n_test: {} \n epochs: {} \n".format(args.task, args.dataset, args.model, args.reg, n_features, n_states, n_train, n_test, args.epochs)
    info += "|".join(["-"] * 80) + "\n"
    return info


def init_logdir(args, infostring, cv=None, iter_oracle=None, warmstart=1):
    args.logdir = get_logdir(args, cv=cv, iter_oracle=iter_oracle, warmstart=warmstart)
    os.makedirs(args.logdir)
    print(f"Logging in {args.logdir}")

    # write important informations in the log directory
    with open(args.logdir + '/parameters.txt', 'w') as file:
        file.write(infostring)
        file.write('\n')
        for arg in vars(args):
            file.write("{}:{}\n".format(arg, getattr(args, arg)))