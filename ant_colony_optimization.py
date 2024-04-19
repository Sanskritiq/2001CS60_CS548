"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2019]-"Text feature selection using ant colony optimization"

run with : `python3 runner.py --algo "aco"`
"""

import numpy as np
from fitness_function import fitness_function

def ant_colony_optimization(feat, label, feat_val, label_val, opts):
    # Parameters
    tau = 1      # pheromone value
    eta = 1      # heuristic desirability
    alpha = 1    # control pheromone
    beta = 0.1   # control heuristic
    rho = 0.2    # pheromone trail decay coefficient

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'tau' in opts:
        tau = opts['tau']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'beta' in opts:
        beta = opts['beta']
    if 'rho' in opts:
        rho = opts['rho']
    if 'eta' in opts:
        eta = opts['eta']

    # Objective function
    fun = fitness_function
    # Number of dimensions
    dim = feat.shape[1]
    # Initial Tau & Eta
    tau = tau * np.ones((dim, dim))
    eta = eta * np.ones((dim, dim))

    fitG = np.inf
    curve = np.zeros(max_Iter)

    t = 1
    # Iterations
    while t <= max_Iter:
        # Reset ant
        X = np.zeros((N, dim))
        for i in range(N):
            # Random number of features
            num_feat = np.random.randint(1, dim + 1)
            # Ant start with random position
            X[i, 0] = np.random.randint(1, dim + 1)
            k = np.array([])
            if num_feat > 1:
                for d in range(1, num_feat):
                    # Start with previous tour
                    k = np.append(k, X[i, d - 1])
                    # Edge/Probability Selection
                    P = (tau[int(k[-1]) - 1, :] ** alpha) * (eta[int(k[-1]) - 1, :] ** beta)
                    # Set selected position = 0 probability
                    P[k.astype(int)-1] = 0
                    # Convert probability
                    prob = P / np.sum(P)
                    # Roulette Wheel selection
                    route = roulette_wheel_selection(prob)
                    # Store selected position to be next tour
                    X[i, d] = route

        # Binary
        X_bin = np.zeros((N, dim))
        for i in range(N):
            # Binary form
            ind = X[i, :].astype(int)
            ind = ind[ind != 0] - 1
            X_bin[i, ind] = 1

        # Fitness
        fit = np.array([fun(feat, label, feat_val, label_val, X_bin[i, :], opts) for i in range(N)])

        # Global update
        best_idx = np.argmin(fit)
        if fit[best_idx] < fitG:
            fitG = fit[best_idx]
            Xgb = X[best_idx, :]

        # Pheromone update
        tauK = np.zeros((dim, dim))
        for i in range(N):
            tour = X[i, :].astype(int)
            tour = tour[tour != 0] - 1
            len_x = len(tour)
            tour = np.append(tour, tour[0])
            for d in range(len_x):
                x = tour[d]
                y = tour[d + 1]
                tauK[x, y] += 1 / (1 + fit[i])

        tauG = np.zeros((dim, dim))
        tour = Xgb.astype(int)
        tour = tour[tour != 0] - 1
        len_g = len(tour)
        tour = np.append(tour, tour[0])
        for d in range(len_g):
            x = tour[d]
            y = tour[d + 1]
            tauG[x, y] = 1 / (1 + fitG)

        # Evaporate pheromone
        tau = (1 - rho) * tau + tauK + tauG

        # Save
        curve[t - 1] = fitG
        print('Iteration', t, 'Best (ACO)=', curve[t - 1])
        t += 1

    # Select features based on selected index
    Sf = Xgb.astype(int)
    Sf = Sf[Sf != 0] - 1
    sFeat = feat[:, Sf]

    # Store results
    ACO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return ACO


# Roulette Wheel Selection
def roulette_wheel_selection(prob):
    # Cumulative summation
    C = np.cumsum(prob)
    # Random one value, most probability value [0~1]
    P = np.random.rand()
    # Roulette wheel
    for i in range(len(C)):
        if C[i] > P:
            return i + 1

def optim(data, labels, data_val, labels_val, opts):
    return ant_colony_optimization(data, labels, data_val, labels_val, opts)