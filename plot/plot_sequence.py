import os
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

base_load = "/home/anowak/INRIA/struntho/logs/"

###############################################################################
#  open m4n
###############################################################################

C4 = [0.1, 0.3, 0.5, 0.7]
data = pickle.load(open(os.path.join(base_load, "m4n_0.1.pkl"), "rb"))
m4ntr1 = data["train_error_batch"]
m4nte1 = data["test_error_batch"]
or1 = data["oracle_error"]

data = pickle.load(open(os.path.join(base_load, "m4n_0.3.pkl"), "rb"))
m4ntr2 = data["train_error_batch"]
m4nte2 = data["test_error_batch"]
or2 = data["oracle_error"]

data = pickle.load(open(os.path.join(base_load, "m4n_0.5.pkl"), "rb"))
m4ntr3 = data["train_error_batch"]
m4nte3 = data["test_error_batch"]
or3 = data["oracle_error"]

data = pickle.load(open(os.path.join(base_load, "m4n_0.7.pkl"), "rb"))
m4ntr4 = data["train_error_batch"]
m4nte4 = data["test_error_batch"]
or4 = data["oracle_error"]

plt.figure(1)
plt.subplot(1, 3, 1)
plt.style.use(['seaborn-darkgrid'])
plt.plot(m4ntr1, label="M4N C = {}".format(C4[0]))
plt.plot(m4ntr2, label="M4N C = {}".format(C4[1]))
plt.plot(m4ntr3, label="M4N C = {}".format(C4[2]))
plt.plot(m4ntr4, label="M4N C = {}".format(C4[3]))
plt.legend(fontsize=14)
plt.title("train error", fontsize=14)
plt.subplot(1, 3, 2)
plt.style.use(['seaborn-darkgrid'])
plt.plot(m4nte1, label="M4N C = {}".format(C4[0]))
plt.plot(m4nte2, label="M4N C = {}".format(C4[1]))
plt.plot(m4nte3, label="M4N C = {}".format(C4[2]))
plt.plot(m4nte4, label="M4N C = {}".format(C4[3]))
plt.legend(fontsize=14)
plt.title("test error", fontsize=14)
plt.subplot(1, 3, 3)
plt.plot(or1, label="M4N C = {}".format(C4[0]))
plt.plot(or2, label="M4N C = {}".format(C4[1]))
plt.plot(or3, label="M4N C = {}".format(C4[2]))
plt.plot(or4, label="M4N C = {}".format(C4[3]))
plt.legend(fontsize=14)
plt.title("oracle", fontsize=14)
plt.show()

C3 = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

data = pickle.load(open(os.path.join(base_load, "m3n_0.005.pkl"), "rb"))
m3ntr00 = data["train_error_batch"]
m3nte00 = data["test_error_batch"]

data = pickle.load(open(os.path.join(base_load, "m3n_0.01.pkl"), "rb"))
m3ntr0 = data["train_error_batch"]
m3nte0 = data["test_error_batch"]


data = pickle.load(open(os.path.join(base_load, "m3n_0.05.pkl"), "rb"))
m3ntr1 = data["train_error_batch"]
m3nte1 = data["test_error_batch"]

data = pickle.load(open(os.path.join(base_load, "m3n_0.1.pkl"), "rb"))
m3ntr2 = data["train_error_batch"]
m3nte2 = data["test_error_batch"]

data = pickle.load(open(os.path.join(base_load, "m3n_0.5.pkl"), "rb"))
m3ntr3 = data["train_error_batch"]
m3nte3 = data["test_error_batch"]

data = pickle.load(open(os.path.join(base_load, "m3n_1.0.pkl"), "rb"))
m3ntr4 = data["train_error_batch"]
m3nte4 = data["test_error_batch"]


plt.figure(2)
plt.subplot(1, 2, 1)
plt.style.use(['seaborn-darkgrid'])
plt.plot(m3ntr00, label="M3N C = {}".format(C3[0]))
plt.plot(m3ntr0, label="M3N C = {}".format(C3[1]))
plt.plot(m3ntr1, label="M3N C = {}".format(C3[2]))
plt.plot(m3ntr2, label="M3N C = {}".format(C3[3]))
plt.plot(m3ntr3, label="M3N C = {}".format(C3[4]))
plt.plot(m3ntr4, label="M3N C = {}".format(C3[5]))
plt.legend(fontsize=14)
plt.title("train error", fontsize=14)
plt.subplot(1, 2, 2)
plt.style.use(['seaborn-darkgrid'])
plt.plot(m3nte00, label="M3N C = {}".format(C3[0]))
plt.plot(m3nte0, label="M3N C = {}".format(C3[1]))
plt.plot(m3nte1, label="M3N C = {}".format(C3[2]))
plt.plot(m3nte2, label="M3N C = {}".format(C3[3]))
plt.plot(m3nte3, label="M3N C = {}".format(C3[4]))
plt.plot(m3nte4, label="M3N C = {}".format(C3[5]))
plt.legend(fontsize=14)
plt.title("test error", fontsize=14)
plt.show()

plt.figure(3)
plt.subplot(1, 2, 1)
plt.style.use(['seaborn-darkgrid'])
plt.plot(m3ntr1, label="M3N C = {}".format(C3[0]))
plt.plot(m4ntr2, label="M4N C = {}".format(C4[1]))
plt.legend(fontsize=14)
plt.title("train error", fontsize=14)
plt.subplot(1, 2, 2)
plt.style.use(['seaborn-darkgrid'])
plt.plot(m3nte1, label="M3N C = {}".format(C3[0]))
plt.plot(m4nte2, label="M4N C = {}".format(C4[1]))
plt.legend(fontsize=14)
plt.title("test error", fontsize=14)
plt.show()