"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2019]-"A new meta-heuristic optimizer: Pathfinder algorithm"

run with : `python3 runner.py --algo "pfa"`
"""

import numpy as np
from fitness_function import fitness_function

def jPathFinderAlgorithm(feat, label,  feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.rand(N, dim) * (ub - lb) + lb

    # Fitness
    fit = np.zeros(N)
    fitP = np.inf

    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Pathfinder update
        if fit[i] < fitP:
            fitP = fit[i]
            Xpf = X[i, :]

    # Set previous pathfinder
    Xpf_old = Xpf

    curve = np.zeros(max_Iter + 1)
    Xpf_new = np.zeros(dim)
    curve[0] = fitP
    t = 1

    # Iterations
    while t <= max_Iter:
        # Alpha & beta in [1,2]
        alpha = 1 + np.random.rand()
        beta = 1 + np.random.rand()

        for d in range(dim):
            # Define u2 in [-1,1]
            u2 = -1 + 2 * np.random.rand()

            # Compute A
            A = u2 * np.exp(-(2 * t) / max_Iter)

            # Update pathfinder
            r3 = np.random.rand()
            Xpf_new[d] = Xpf[d] + 2 * r3 * (Xpf[d] - Xpf_old[d]) + A

        # Boundary
        Xpf_new = np.clip(Xpf_new, lb, ub)

        # Update previous path
        Xpf_old = Xpf.copy()

        # Fitness
        Fnew = fun(feat, label, (Xpf_new > thres), opts)

        # Greedy selection
        if Fnew < fitP:
            fitP = Fnew
            Xpf = Xpf_new

        # Sort member
        idx = np.argsort(fit)
        fit = fit[idx]
        X = X[idx, :]

        # Update first solution
        if Fnew < fit[0]:
            fit[0] = Fnew
            X[0, :] = Xpf_new

        # Update
        for i in range(1, N):
            # Distance
            Dij = np.linalg.norm(X[i, :] - X[i - 1, :])

            for d in range(dim):
                # Define u1 in [-1,1]
                u1 = -1 + 2 * np.random.rand()

                # Compute epsilon
                eps = (1 - (t / max_Iter)) * u1 * Dij

                # Define R1, R2
                r1 = np.random.rand()
                r2 = np.random.rand()
                R1 = alpha * r1
                R2 = beta * r2

                # Update member
                X[i, d] = X[i, d] + R1 * (X[i - 1, d] - X[i, d]) + R2 * (Xpf[d] - X[i, d]) + eps

            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        # Fitness
        for i in range(1, N):
            # Fitness
            Fnew = fun(feat, label, (X[i, :] > thres), opts)

            # Selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i, :] = X[i, :]

            # Pathfinder update
            if fit[i] < fitP:
                fitP = fit[i]
                Xpf = X[i, :]

        curve[t] = fitP
        print('Iteration %d Best (PFA)= %f' % (t, curve[t]))
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xpf > thres]
    sFeat = feat[:, Sf]

    # Store results
    PFA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return PFA

def optim(feat, label, feat_val, label_val, opts):
    return jPathFinderAlgorithm(feat, label, feat_val, label_val, opts)
