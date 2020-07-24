import os
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from terminaltables import AsciiTable

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

base_path = "/home/anowak/struntho/logs/"
task = "multiclass"
# task = "ordinal"
# task = "sequence"
# task = "ranking"

if task == "multiclass":
    Datasets = ["iris", "satimage", "segment", "vehicle", "wine", "letter", "krvsk", "mfeatfactors"]
elif task == "ordinal":
    Datasets = ["wisconsin", "stocks", "machinecpu", "abalone", "auto"]
elif task == "sequence":
    Datasets = ["ocr"] #, "conll"] #, "ner", "conll"]
elif task == "ranking":
    Datasets = ["glass", "bodyfat", "authorship", "wine", "vowel", "vehicle"]
else:
    raise ValueError("task {} not implemented.".format(task))

Models = ["m3n", "crf", "m4n"]
# Models = ["m3n", "crf"] #, "m4n"]
Models = ["m3n", "m4n"]

cross_valid = {"m3n": {}, "crf": {}, "m4n": {}}
cross_valid_regs = {"m3n": {}, "crf": {}, "m4n": {}}
# cross_valid = {"m3n": {}, "crf": {}} #, "m4n": {}}
# cross_valid_regs = {"m3n": {}, "crf": {}} #, "m4n": {}}
cross_valid = {"m3n": {}, "m4n": {}}
cross_valid_regs = {"m3n": {}, "m4n": {}}

###############################################################################
# plot all experiments and compute cross validation
###############################################################################

for dataset in Datasets:

    for model in Models:
        test_loss = np.inf

        path_datasets = os.path.join(base_path, "{}_{}_{}".format(task, model, dataset))
        dirs = [dI for dI in os.listdir(path_datasets) if os.path.isdir(os.path.join(path_datasets,dI)) and dI[0] != "_" and len(dI) == 32]

        # keep only the last ones
        dirs_dic = {}
        for d in dirs:
            # if model == "m4n":
            #     pdb.set_trace()
            reg = float(d[16:28])
            if reg in dirs_dic.keys() and int(d[-1]) in dirs_dic[reg]:
                dirs_dic[reg][int(d[-1])].append(d)
            elif reg in dirs_dic.keys():
                dirs_dic[reg][int(d[-1])] = [d]
            else:
                dirs_dic[reg] = {}
                dirs_dic[reg][int(d[-1])] = [d]
        # if dataset == "pos":
        #     pdb.set_trace()
        # choose the newest experiments
        todel = []
        for i, reg in enumerate(dirs_dic.keys()):
            # pdb.set_trace()
            if len(dirs_dic[reg]) == 4:
                for cv in range(4):
                    # pdb.set_trace()
                    try:
                        dirs_dic[reg][cv] = sorted(dirs_dic[reg][cv])[-1]
                        print(dirs_dic[reg][cv])
                    except:
                        pdb.set_trace()
            else:
                todel.append(reg)
        for reg in todel:
            del dirs_dic[reg]
                

        for i, reg in enumerate(dirs_dic.keys()):
            # choose the best in the cross validation
            test_error_finals = []
            for cv in range(4):
                # pdb.set_trace()
                path_exp = os.path.join(path_datasets, dirs_dic[reg][cv])
                results = pickle.load(open(os.path.join(path_exp, "objectives.pkl"), "rb"))
                test_error_finals.append(results["test_error_final"])
            test_error_finals = np.array(test_error_finals)
            mean, std = test_error_finals.mean(), test_error_finals.std()
            # if dataset == "machinecpu":
            #     pdb.set_trace()
                
            if mean < test_loss:
                cross_valid[model][dataset] = [results, mean, std]
                cross_valid_regs[model][dataset] = reg
                test_loss = mean

###############################################################################
# print cross validation table
###############################################################################


cross_valid_info = [["", "model", "lambda", "test error", "std", "train error", "primal", "dual", "dual gap", "epochs"]]
for dataset in Datasets:
    for j, model in enumerate(Models):
        info = ["", model]
        if j == 0: info[0] = dataset            
        reg = cross_valid_regs[model][dataset]
        [results, mean, std] = cross_valid[model][dataset] 
        info = info + ["{:.2E}".format(reg), 
                        # "{:.2E}".format(results["test loss"][-1]),
                        "{:.2E}".format(mean),
                        "{:.2E}".format(std),
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

# plt.figure(figsize=(20,20))
# plt.style.use(['seaborn-darkgrid'])
# for i, dataset in enumerate(Datasets):
#     plt.subplot(2, 3, i+1)
#     plt.title(dataset)
#     for j, model in enumerate(Models):
#         results = cross_valid[model][dataset]
#         test_error = results["test loss"]
#         iteration = results["iteration"]
#         plt.plot(iteration, test_error)
#         plt.ylabel("test loss")
#         plt.xlabel("iterations")
# plt.show()
