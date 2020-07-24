import os
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from terminaltables import AsciiTable

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

base_path = "/home/anowak/struntho/logs/"
# task = "multiclass"
task = "ordinal"
# task = "sequence"
# task = "ranking"
model = "m4n"

if task == "multiclass":
    Datasets = ["iris", "satimage", "segment", "vehicle", "wine", "letter", "krvsk", "mfeatfactors"]
elif task == "ordinal":
    Datasets = ["wisconsin", "stocks", "machinecpu", "auto"] # "abalone", ]
elif task == "sequence":
    Datasets = ["ocr"] #, "conll"] #, "ner", "conll"]
elif task == "ranking":
    Datasets = ["glass", "bodyfat", "authorship", "wine", "vowel", "vehicle"]
else:
    raise ValueError("task {} not implemented.".format(task))

###############################################################################
# plot all experiments and compute cross validation
###############################################################################
mp = {5: 0, 10: 1, 20: 2, 50: 3, 100: 4}

for dataset in Datasets:
    path_datasets = os.path.join(base_path, "{}_{}_{}".format(task, model, dataset))
    dirs = [dI for dI in os.listdir(path_datasets) if os.path.isdir(os.path.join(path_datasets,dI)) and dI[0] != "_" and len(dI) == 45]

    # keep only the last ones
    dirs_dic = {0: [0]*5, 1: [0]*5}
    for d in dirs:
        w = int(d[29])
        iter_oracle = int(d[-3:])
        if iter_oracle in mp.keys():
            if dirs_dic[w][mp[iter_oracle]] == 0:
                dirs_dic[w][mp[iter_oracle]] = [d]
            else:
                dirs_dic[w][mp[iter_oracle]].append(d)

    for w in range(2):
        for i in range(5):
            dirs_dic[w][i] = sorted(dirs_dic[w][i])[-1]

    curves = {0: [0]*5, 1: [0]*5}
    for w in range(2):
        for i in range(5):
            path_exp = os.path.join(path_datasets, dirs_dic[w][i])
            try: 
                results = pickle.load(open(os.path.join(path_exp, "objectives.pkl"), "rb"))
            except:
                pdb.set_trace()
            curves[w][i] = [results["test loss"], results["dual gap"], results["oracles"], results["dual objective"], results["iteration"]]

    # plots
    # plt.figure(figsize=(20,20))
    # plt.style.use(['seaborn-darkgrid'])
    # plt.title(dataset)
    for w in range(2):
        for i in range(5):
            results = curves[w][i]
            # pdb.set_trace()
            test_loss, dual_gap, oracle, dual, iteration = results
            print("{} {} {}".format(dataset, w, i))
            print("test loss {}, dual {}".format(np.min(test_loss), np.min(oracle)))
            # pdb.set_trace()
            # iteration = list(np.arange(21))
            # oracle = [oracle[it * 10] for it in iteration]
            # plt.subplot(2, 3, w * 3 + 1)
            # plt.plot(iteration, test_loss, c=colors[i % 8])
            # plt.title("test loss")
            # plt.xlabel("passes data")
            # plt.subplot(2, 3, w * 3 + 2)
            # plt.plot(iteration, dual, c=colors[i % 8])
            # plt.title("dual objective")
            # plt.xlabel("passes data")
            # plt.subplot(2, 3, w * 3 + 3)
            # plt.plot(iteration, oracle, c=colors[i % 8])
            # plt.title("oracle error")
            # plt.xlabel("passes data")
    # plt.show()
