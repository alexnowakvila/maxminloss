import pickle
import time
from .arguments import get_logdir
import numpy as np
from pathlib import Path

class Logger:

    def __init__(self, args, cv=None, iter_oracle=None, warmstart=1):

        self.args = args
        self.logdir = get_logdir(args, cv=cv, iter_oracle=iter_oracle, warmstart=warmstart)

        self.delta_time = time.time()

        # general
        self.train_error = -1
        self.test_error = -1
        self.iteration = -1
        self.train_error_batch = -1
        self.test_error_batch = -1
        self.iteration_batch = -1
        self.primal_objective = -1
        self.dual_objective = -1
        self.dual_gap = -1
        self.oracles = [-1]

        # save values in a dictionary
        self.results = {
            "train loss": [],
            "test loss": [],
            "iteration": [],
            "train loss batch": [],
            "test loss batch": [],
            "iteration batch": [],
            "primal objective": [],
            "dual objective": [],
            "dual gap": [],
            "oracles": [],
            "time": None,
            "test_error_final": -1
        }

    def __repr__(self):
        string = "epoch {} | reg {:.2E} | train loss {:.4f} | test loss {:.4f} | primal {:.6f} | dual {:.6f} | dual gap {:.6f} | error oracles {:.4f} \n ".format(
            self.iteration,
            self.args.reg,
            self.train_error,
            self.test_error,
            self.primal_objective,
            self.dual_objective,
            self.dual_gap,
            self.oracles[-1]
        )
        return string

    def append_results_batch(self):
        self.results["train loss batch"].append(self.train_error_batch)
        self.results["test loss batch"].append(self.test_error_batch)
        self.results["iteration batch"].append(self.iteration_batch)
        self.results["time"] = time.time() - self.delta_time 

    def append_results(self):
        self.results["train loss"].append(self.train_error)
        self.results["test loss"].append(self.test_error)
        self.results["iteration"].append(self.iteration)
        self.results["primal objective"].append(self.primal_objective)
        self.results["dual objective"].append(self.dual_objective)
        self.results["dual gap"].append(self.dual_gap)
        self.results["oracles"] = self.oracles
        self.results["time"] = time.time() - self.delta_time

    def save_results(self):
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        with open(self.logdir + "/objectives.pkl", "wb") as out:
            pickle.dump(self.results, out)