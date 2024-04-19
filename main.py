import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from fs import fs  # assuming you have a wrapper for the feature selection methods

# Load dataset
data = pd.read_csv('NF-BOT-IOT_train_preprocessed.csv').values
feat = data[:, :-1]
label = data[:, -1]

# Common parameter settings
opts = {
    'k': 5,  # Number of k in K-nearest neighbor
    'N': 10,  # Number of solutions
    'T': 100,  # Maximum number of iterations
}

# Ratio of validation data
ho = 0.2

# Divide data into training and validation sets
feat_train, feat_val, label_train, label_val = train_test_split(feat, label, test_size=ho)

# Perform feature selection with Particle Swarm Optimization (PSO)
# opts_pso = {
#     **opts,
#     'c1': 2,
#     'c2': 2,
#     'w': 0.9,
# }
# FS_pso = fs('pso', feat_train, label_train, opts_pso)

opts_abc = {
    **opts,
    'max': 5,
}
FS_abc = fs('abc', feat_train, label_train, feat_val, label_val, opts_abc)

# Define index of selected features
sf_idx_pso = FS_abc['sf']

# Accuracy
knn_pso = KNeighborsClassifier(n_neighbors=opts['k'])
knn_pso.fit(feat_train[:, sf_idx_pso], label_train)
acc_pso = accuracy_score(label_val, knn_pso.predict(feat_val[:, sf_idx_pso]))

# Plot convergence
plt.plot(FS_abc['c'])
plt.xlabel('Number of Iterations')
plt.ylabel('Fitness Value')
plt.title('abc')
plt.grid(True)
plt.show()

# Repeat the above steps for other feature selection methods (SMA, WOA) with corresponding parameters
# You can create separate functions for repeating parts to avoid redundancy
