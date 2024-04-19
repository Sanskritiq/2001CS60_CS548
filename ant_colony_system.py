"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2006]-"Ant Colony Optimization"

run with : `python3 runner.py --algo "acs"`
"""

import numpy as np
from fitness_function import fitness_function

def ant_colony_system(feat, label, feat_val, label_val, opts):
    # Parameters
    tau = 1  # pheromone value
    eta = 1  # heuristic desirability
    alpha = 1  # control pheromone
    beta = 1  # control heuristic
    rho = 0.2  # pheromone trail decay coefficient
    phi = 0.5  # pheromena coefficient

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
    if 'phi' in opts:
        phi = opts['phi']

    # Objective function
    fun = fitness_function

    # Number of dimensions
    dim = feat.shape[1]

    # Initial Tau & Eta
    tau = tau * np.ones((dim, dim))
    eta = eta * np.ones((dim, dim))
    tau0 = tau.copy()

    # Pre
    fitG = np.inf
    fit = np.zeros(N)

    curve = np.zeros(max_Iter)
    t = 1

    # Iterations
    while t <= max_Iter:
        # Reset ant
        X = np.zeros((N, dim))
        for i in range(N):
            # Set number of features
            num_feat = np.random.randint(1, dim + 1)
            # Ant start with random position
            X[i, 0] = np.random.randint(1, dim + 1)
            k = np.array([])
            if num_feat > 1:
                for d in range(1, num_feat):
                    # Start with previous tour
                    k = np.append(k, X[i, d - 1])
                    # Edge / Probability Selection
                    P = (tau[int(k[-1]) - 1, :] ** alpha) * (eta[int(k[-1]) - 1, :] ** beta)
                    # Set selected position = 0 probability
                    P[k.astype(int) - 1] = 0
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
            ind = X[i, :].astype(int) - 1
            X_bin[i, ind] = 1

        # Fitness
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, feat_val, label_val, X_bin[i, :], opts)
            # Global update
            if fit[i] < fitG:
                Xgb = X[i, :]
                fitG = fit[i]

        # Tau update
        tour = Xgb.astype(int)
        tour = np.delete(tour, np.where(tour == 0))
        tour = np.append(tour, tour[0])
        for d in range(len(tour) - 1):
            # Feature selected
            x = tour[d] - 1
            y = tour[d + 1] - 1
            # Delta tau
            Dtau = 1 / fitG
            # Update tau
            tau[x, y] = (1 - phi) * tau[x, y] + phi * Dtau

        # Evaporate pheromone
        tau = (1 - rho) * tau + rho * tau0

        # Save
        curve[t - 1] = fitG
        print('Iteration', t, 'Best (ACS)=', curve[t - 1])
        t += 1

    # Select features based on selected index
    Sf = np.unique(Xgb.astype(int))
    Sf = Sf[Sf > 0] - 1
    sFeat = feat[:, Sf]

    # Store results
    ACS = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return ACS


def roulette_wheel_selection(prob):
    # Cumulative summation
    C = np.cumsum(prob)
    # Random one value, most probable value [0~1]
    P = np.random.rand()
    # Roulette wheel
    for i in range(len(C)):
        if C[i] > P:
            return i + 1

    return len(C)  # In case no value is selected

def optim(feat, label, feat_val, label_val, opts):
    return ant_colony_system(feat, label, feat_val, label_val, opts)
