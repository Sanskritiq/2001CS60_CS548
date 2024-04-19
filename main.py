import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from fs import fs  # assuming you have a wrapper for the feature selection methods

# Load dataset
train_data = pd.read_csv('NF-BOT-IOT_train_preprocessed.csv').values
test_data = pd.read_csv('NF-BOT-IOT_test_preprocessed.csv').values

# Common parameter settings
opts = {
    'k': 5,  # Number of k in K-nearest neighbor
    'N': 10,  # Number of solutions
    'T': 100,  # Maximum number of iterations
}

# Divide data into training and validation sets
feat_train, feat_val, label_train, label_val = train_data[:, :-1], test_data[:, :-1], train_data[:, -1], test_data[:, -1]

# Perform feature selection with Particle Swarm Optimization (PSO)
# opts_pso = {
#     **opts,
#     'c1': 2,
#     'c2': 2,
#     'w': 0.9,
# }
# FS_pso = fs('pso', feat_train, label_train, opts_pso)

opts_acd = {
    **opts,
    'tau': 1,
    'eta': 1,
    'alpha': 1,
    'beta': 0.1,
    'rho': 0.2,
}
opts_acs = {
    **opts,
    'tau': 1,
    'eta': 1,
    'alpha': 1,
    'beta': 0.1,
    'rho': 0.2,
    'phi': 0.5,
}
opts_abc = {
    **opts,
    'max': 5,
}
opts_abo = {
    **opts,
    'stepe': 0.05,
    'ratio': 0.2,
    'ty': 1,
}
opts_aso = {
    **opts,
    'alpha': 50,
    'beta': 0.2,
}
opts_ba = {
    **opts,
    'fmax': 2,
    'fmin': 0,
    'alpha': 0.9,
    'gamma': 0.9,
    'A': 2,
    'r': 1,
}
opts_boa = {
    **opts,
    'c': 0.01,
    'p': 0.8,
}

opts_csa = {
    **opts,
    'AP': 0.1,
    'fl': 1.5,
}

opts_cs = {
    **opts,
    'Pa': 0.25,
}

opts_de = {
    **opts,
    'CR': 0.9,
    'F': 0.5,
}

opts_epo = {
    **opts,
    'M': 2,
    'f': 3,
    'l': 2,
}

opts_eo = {
    **opts,
    'a1': 2,
    'a2': 1,
    'GP': 0.5,
}

opts_fa = {
    **opts,
    'alpha': 1,
    'beta0': 1,
    'gamma': 1,
    'theta': 0.97,
}

opts_fpa = {
    **opts,
    'P': 0.8,
}

opts_ga = {
    **opts,
    'CR': 0.8,
    'MR': 0.01,
    'Ts': 3,  # Only for 'gat'
}

opts_gsa = {
    **opts,
    'G0': 100,
    'alpha': 20,
}

opts_hs = {
    **opts,
    'PAR': 0.05,
    'HMCR': 0.7,
    'bw': 0.2,
}

opts_hgso = {
    **opts,
    'Nc': 2,
    'K': 1,
    'alpha': 1,
    'beta': 1,
    'L1': 5E-3,
    'L2': 100,
    'L3': 1E-2,
}

opts_hlo = {
    **opts,
    'pi': 0.85,
    'pr': 0.1,
}

opts_mrfo = {
    **opts,
    'S': 2,
}

opts_mpa = {
    **opts,
    'P': 0.5,
    'FADs': 0.2,
}

opts_mbo = {
    **opts,
    'peri': 1.2,
    'p': 5/12,
    'Smax': 1,
    'BAR': 5/12,
    'N1': 4,
}

opts_mfo = {
    **opts,
    'b': 1,
}

opts_mvo = {
    **opts,
    'p': 6,
    'Wmax': 1,
    'Wmin': 0.2,
}

opts_pro = {
    **opts,
    'Pmut': 0.06,
}

opts_sbo = {
    **opts,
    'alpha': 0.94,
    'z': 0.02,
    'MR': 0.05,
}

opts_sa = {
    **opts,
    'c': 0.93,
    'T0': 100,
}

opts_sca = {
    **opts,
    'alpha': 2,
}

opts_tga = {
    **opts,
    'N1': 3,
    'N2': 5,
    'N4': 3,
    'theta': 0.8,
    'lambda': 0.5,
}

opts_tsa = {
    **opts,
    'ST': 0.1,
}

opts_wsa = {
    **opts,
    'tau': 0.8,
    'sl': 0.035,
    'phi': 0.001,
    'lambda': 0.75,
}

opts_woa = {
    **opts,
    'b': 1,
}

optimization_algorithms = [
    'acd', 'acs', 'abc', 'abo', 'aso', 'ba', 'boa', 'csa', 'cs', 'de', 'epo',
    'eo', 'fa', 'fpa', 'ga', 'gsa', 'hs', 'hgso', 'hlo', 'mrfo', 'mpa', 'mbo',
    'mfo', 'mvo', 'pro', 'sbo', 'sa', 'sca', 'tga', 'tsa', 'wsa', 'woa'
]

FS = fs('ba', feat_train, label_train, feat_val, label_val, opts_ba)

# Define index of selected features
sf_idx = FS['sf']

# Accuracy
knn = KNeighborsClassifier(n_neighbors=opts['k'])
knn.fit(feat_train[:, sf_idx], label_train)
acc = accuracy_score(label_val, knn.predict(feat_val[:, sf_idx]))

# Plot convergence
plt.plot(FS['c'])
plt.xlabel('Number of Iterations')
plt.ylabel('Fitness Value')
plt.title('abc')
plt.grid(True)
plt.show()

# Repeat the above steps for other feature selection methods (SMA, WOA) with corresponding parameters
# You can create separate functions for repeating parts to avoid redundancy
