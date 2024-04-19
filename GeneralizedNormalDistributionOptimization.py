"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2020]-"Generalized normal distribution optimization and its 
applications in parameter extraction of photovoltaic models"

run with : `python3 runner.py --algo "gndo"`
"""
from fitness_function import fitness_function
import numpy as np


def j_generalized_normal_distribution_optimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(X, opts):
        return fitness_function(feat, label, feat_val, label_val, X, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(X[i], opts)
        # Best
        if fit[i] < fitG:
            fitG = fit[i]
            Xb = X[i]

    # Pre
    V = np.zeros((N, dim))

    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    t = 1

    # Iteration
    while t <= max_Iter:
        # Compute mean position
        M = np.mean(X, axis=0)
        for i in range(N):
            alpha = np.random.rand()
            # Local exploitation
            if alpha > 0.5:
                # Random numbers
                a = np.random.rand()
                b = np.random.rand()
                for d in range(dim):
                    # Compute mean
                    mu = (1/3) * (X[i, d] + Xb[d] + M[d])
                    # Compute standard deviation
                    delta = np.sqrt((1/3) * ((X[i, d] - mu) ** 2 + (Xb[d] - mu) ** 2 + (M[d] - mu) ** 2))
                    # Compute eta
                    lambda1 = np.random.rand()
                    lambda2 = np.random.rand()
                    if a <= b:
                        eta = np.sqrt(-1 * np.log(lambda1)) * np.cos(2 * np.pi * lambda2)
                    else:
                        eta = np.sqrt(-1 * np.log(lambda1)) * np.cos(2 * np.pi * lambda2 + np.pi)
                    # Generate normal distribution
                    V[i, d] = mu + delta * eta
            # Global Exploitation
            else:
                # Random three vectors but not i
                RN = np.random.permutation(N)
                RN = RN[RN != i]
                p1 = RN[0]
                p2 = RN[1]
                p3 = RN[2]
                # Random beta
                beta = np.random.rand()
                # Normal random number: zero mean & unit variance
                lambda3 = np.random.randn()
                lambda4 = np.random.randn()
                # Get v1
                v1 = X[i] - X[p1] if fit[i] < fit[p1] else X[p1] - X[i]
                # Get v2
                v2 = X[p2] - X[p3] if fit[p2] < fit[p3] else X[p3] - X[p2]
                # Generate new position
                for d in range(dim):
                    V[i, d] = X[i, d] + beta * (np.abs(lambda3) * v1[d]) + (1 - beta) * (np.abs(lambda4) * v2[d])
            # Boundary
            V[i] = np.clip(V[i], lb, ub)

        # Fitness
        for i in range(N):
            fitV = fun(V[i], opts)
            # Greedy selection
            if fitV < fit[i]:
                fit[i] = fitV
                X[i] = V[i]
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xb = X[i]

        # Save
        curve[t] = fitG
        print('Iteration {} Best (GNDO)= {}'.format(t, curve[t]))
        t += 1

    # Select features
    Sf = np.where(Xb > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    GNDO = {'sf': Sf.tolist(), 'ff': sFeat.tolist(), 'nf': len(Sf), 'c': curve.tolist(), 'f': feat.tolist(),
            'l': label.tolist()}
    return GNDO


def optim(feat, label, feat_val, label_val, opts):
    return j_generalized_normal_distribution_optimization(feat, label, feat_val, label_val, opts)