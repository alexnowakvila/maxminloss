import os
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from terminaltables import AsciiTable

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

base_path = "/home/anowak/struntho/logs/"

Datasets = ["iris", "satimage", "segment", "vehicle", "wine"]
Models = ["m3n", "crf", "m4n"]

cross_valid = {"m3n": {}, "crf": {}, "m4n": {}}
cross_valid_regs = {"m3n": {}, "crf": {}, "m4n": {}}

###############################################################################
# plot all experiments and compute cross validation
###############################################################################

for dataset in Datasets:

    for model in Models:
        test_loss = np.inf

        path_datasets = os.path.join(base_path, "{}_{}".format(model, dataset))
        dirs = [dI for dI in os.listdir(path_datasets) if os.path.isdir(os.path.join(path_datasets,dI)) and dI[0] != "_" and len(dI) == 28]

        # keep only the last ones
        dirs_dic = {}
        for d in dirs:
            # if model == "m4n":
            #     pdb.set_trace()
            reg = float(d[16:28])
            if reg in dirs_dic.keys():
                dirs_dic[reg].append(d)
            else:
                dirs_dic[reg] = [d]
        plt.figure(figsize=(20,20))
        plt.style.use(['seaborn-darkgrid'])
        for i, reg in enumerate(dirs_dic.keys()):
            dirs_dic[reg] =sorted(dirs_dic[reg])[-1]
            
            # load data
            path_exp = os.path.join(path_datasets, dirs_dic[reg])
            results = pickle.load(open(os.path.join(path_exp, "objectives.pkl"), "rb"))

            train_error = results["train loss"]
            test_error = results["test loss"]
            iteration = results["iteration"]
            primal_objective = results["primal objective"]
            dual_objective = results["dual objective"]
            dual_gap = results["dual gap"]

            if test_error[-1] < test_loss:
                cross_valid[model][dataset] = results
                cross_valid_regs[model][dataset] = reg

            plt.subplot(2, 2, 1)
            plt.title(dataset + ' ' + model + ' train error')
            plt.plot(iteration, train_error, '--', c=colors[i % 8], label=str(reg))
            plt.xlabel("iterations")
            plt.subplot(2, 2, 2)
            plt.title(dataset + ' ' + model + ' test error')
            plt.plot(iteration, test_error, c=colors[i % 8], label=str(reg))
            plt.xlabel("iterations")
            plt.subplot(2, 2, 3)
            plt.title(dataset + ' ' + model + ' dual obj')
            plt.plot(iteration, dual_objective, c=colors[i % 8], label=str(reg))
            plt.xlabel("iterations")
            plt.subplot(2, 2, 4)
            plt.title(dataset + ' ' + model + ' dual gap')
            plt.plot(iteration, dual_gap, c=colors[i % 8], label=str(reg))
            plt.xlabel("iterations")
            plt.legend()
        # plt.show()


###############################################################################
# print cross validation table
###############################################################################


cross_valid_info = [["", "model", "lambda", "test error", "train error", "primal", "dual", "dual gap", "epochs"]]
for dataset in Datasets:
    for j, model in enumerate(Models):
        info = ["", model]
        if j == 0: info[0] = dataset            
        reg = cross_valid_regs[model][dataset]
        results = cross_valid[model][dataset] 
        info = info + ["{:.2E}".format(reg), 
                        "{:.2E}".format(results["test loss"][-1]),
                        "{:.2E}".format(results["train loss"][-1]),
                        "{:.2E}".format(results["primal objective"][-1]),
                        "{:.2E}".format(results["dual objective"][-1]),
                        "{:.2E}".format(results["dual gap"][-1]),
                        results["iteration"][-1]+1]
        cross_valid_info.append(info)

datasets_table = AsciiTable(cross_valid_info)
print(datasets_table.table)


###############################################################################
# plot cross validated experiments
###############################################################################

plt.figure(figsize=(20,20))
plt.style.use(['seaborn-darkgrid'])
for i, dataset in enumerate(Datasets):
    plt.subplot(2, 3, i+1)
    plt.title(dataset)
    for j, model in enumerate(Models):
        results = cross_valid[model][dataset]
        test_error = results["test loss"]
        iteration = results["iteration"]
        plt.plot(iteration, test_error)
        plt.ylabel("test loss")
        plt.xlabel("iterations")
plt.show()
