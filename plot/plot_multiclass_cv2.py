import os
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from terminaltables import AsciiTable
from scipy import stats
from scipy.stats import wilcoxon

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

base_path = "/home/anowak/struntho/logs/"
# task = "multiclass"
task = "ordinal"
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
# Models = ["m3n", "m4n"]

cross_valid = {"m3n": {}, "crf": {}, "m4n": {}}
cross_valid_regs = {"m3n": {}, "crf": {}, "m4n": {}}
# cross_valid = {"m3n": {}, "crf": {}} #, "m4n": {}}
# cross_valid_regs = {"m3n": {}, "crf": {}} #, "m4n": {}}
# cross_valid = {"m3n": {}, "m4n": {}}
# cross_valid_regs = {"m3n": {}, "m4n": {}}

###############################################################################
# plot all experiments and compute cross validation
###############################################################################
CV = 13

for dataset in Datasets:

    for model in Models:
        test_loss = np.inf

        path_datasets = os.path.join(base_path, "{}_{}_{}".format(task, model, dataset))
        dirs = [dI for dI in os.listdir(path_datasets) if os.path.isdir(os.path.join(path_datasets,dI)) and dI[0] != "_" and (len(dI) == 32 or (len(dI) == 33 and int(dI[-1])<3))]

        # keep only the last ones
        dirs_dic = {}
        for d in dirs:
            # if model == "m4n":
            #     pdb.set_trace()
            reg = float(d[16:28])
            cv = int(d[-1])
            if len(d) == 33:
                cv = int(d[-2:])
            if reg in dirs_dic.keys() and cv in dirs_dic[reg]:
                dirs_dic[reg][cv].append(d)
            elif reg in dirs_dic.keys():
                dirs_dic[reg][cv] = [d]
            else:
                dirs_dic[reg] = {}
                dirs_dic[reg][cv] = [d]
        # if dataset == "pos":
        #     pdb.set_trace()
        # choose the newest experiments
        todel = []
        if task == "ranking":
            del dirs_dic[0.0002]
            del dirs_dic[0.0007]
            del dirs_dic[0.002]
            del dirs_dic[0.005]
        for i, reg in enumerate(dirs_dic.keys()):
            # pdb.set_trace()
            # if len(dirs_dic[reg]) == CV:
            for cv in range(CV):
                # pdb.set_trace()
                try:
                    dirs_dic[reg][cv] = sorted(dirs_dic[reg][cv])[-1]
                    # print(dirs_dic[reg][cv])
                except:
                    pdb.set_trace()
            # else:
            #     todel.append(reg)
        # for reg in todel:
        #     pdb.set_trace()
        # del dirs_dic[8e-5]
        # pdb.set_trace()
        for i, reg in enumerate(dirs_dic.keys()):
            # choose the best in the cross validation
            test_error_finals = []
            for cv in range(CV):
                # pdb.set_trace()
                path_exp = os.path.join(path_datasets, dirs_dic[reg][cv])
                results = pickle.load(open(os.path.join(path_exp, "objectives.pkl"), "rb"))
                test_error_finals.append(results["test_error_final"])
            test_error_finals = np.array(test_error_finals)
            # if dataset == "machinecpu":
            #     pdb.set_trace()
            
            mean = test_error_finals.mean()
            if mean < test_loss:
                cross_valid[model][dataset] = [results, test_error_finals]
                cross_valid_regs[model][dataset] = reg
                test_loss = mean

###############################################################################
# run significance test
###############################################################################

differents = {}
for dataset in Datasets:
    Test_Errors = {}
    means = []
    for j, model in enumerate(Models):
        [results, test_error_finals] = cross_valid[model][dataset] 
        Test_Errors[model] = test_error_finals

    try:
        # result = stats.ttest_rel(Test_Errors["m3n"], Test_Errors["m4n"])[1]
        result =  wilcoxon(Test_Errors["m3n"] - Test_Errors["m4n"])[1]
        different = True if result < 0.1 else False
        print("result test is {}".format(different))
    except:
        different = False
    if Test_Errors["m3n"].mean() > Test_Errors["m4n"].mean() and different:
        differents[dataset] = "m4n"
    elif Test_Errors["m4n"].mean() < Test_Errors["m4n"].mean() and different:
        differents[dataset] = "m3n"
    else:
        differents[dataset] = "equal"
    
###############################################################################
# print cross validation table
###############################################################################


cross_valid_info = [["", "test", "model", "test error", "lambda", "std", "train error", "primal", "dual", "dual gap", "epochs"]]
for dataset in Datasets:
    for j, model in enumerate(Models):
        info = ["", "", model]
        if j == 0: 
            info[0] = dataset
            info[1] = differents[dataset]         
        reg = cross_valid_regs[model][dataset]
        [results, test_error_finals] = cross_valid[model][dataset] 
        mean, std = test_error_finals.mean(), test_error_finals.std()
        info = info + ["{:.2E}".format(mean), 
                        # "{:.2E}".format(results["test loss"][-1]),
                        "{:.2E}".format(reg),
                        "{:.2E}".format(std),
                        # "{:01d}".format(different),
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
