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

run with : `python3 runner.py --algo "mvo"`
"""

import numpy as np
from fitness_function import fitness_function


def jMultiVerseOptimizer(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    p = 6  # control TDR
    Wmax = 1  # maximum WEP
    Wmin = 0.2  # minimum WEP
    type = 1

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'p' in opts:
        p = opts['p']
    if 'Wmin' in opts:
        Wmin = opts['Wmin']
    if 'Wmax' in opts:
        Wmax = opts['Wmax']
    if 'ty' in opts:
        type = opts['ty']
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


    curve = np.inf * np.ones(max_Iter+1)
    t = 1

    # Iterations
    while t <= max_Iter:
        # Calculate inflation rate
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best universe
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Sort universe from best to worst
        idx = np.argsort(fit)
        fitSU = fit[idx]
        X_SU = X[idx, :]

        # Elitism (first 1 is elite)
        X[0, :] = X_SU[0, :]

        # Either 1-norm or 2-norm
        if type == 1:
            # Normalize inflation rate using 2-norm
            NI = fitSU / np.sqrt(np.sum(fitSU ** 2))
        elif type == 2:
            # Normalize inflation rate using 1-norm
            NI = fitSU / np.sum(fitSU)

        # Normalize inverse inflation rate using 1-norm
        inv_fitSU = 1 / (1 + fitSU)
        inv_NI = inv_fitSU / np.sum(inv_fitSU)

        # Wormhole Existence probability
        WEP = Wmin + t * ((Wmax - Wmin) / max_Iter)

        # Travelling disrance rate
        TDR = 1 - ((t ** (1 / p)) / (max_Iter ** (1 / p)))

        # Start with 2 since first is elite
        for i in range(1, N):
            # Define black hole
            idx_BH = i
            for d in range(dim):
                # White/black hole tunnels & exchange object of universes
                r1 = np.random.rand()
                if r1 < NI[i]:
                    # Random select k with roulette wheel
                    idx_WH = jRouletteWheelSelection(inv_NI)
                    # Position update
                    X[idx_BH, d] = X_SU[idx_WH, d]

                # Local changes for universes
                r2 = np.random.rand()
                if r2 < WEP:
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    if r3 < 0.5:
                        X[i, d] = Xgb[d] + TDR * ((ub - lb) * r4 + lb)
                    else:
                        X[i, d] = Xgb[d] - TDR * ((ub - lb) * r4 + lb)
                else:
                    X[i, d] = X[i, d]

            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        curve[t] = fitG
        print('Iteration %d Best (MVO)= %f' % (t, curve[t]))
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    MVO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return MVO


def jRouletteWheelSelection(prob):
    # Cumulative summation
    C = np.cumsum(prob)
    # Random one value, most probability value [0~1]
    P = np.random.rand()
    # Route wheel
    for i in range(len(C)):
        if C[i] > P:
            return i
    return len(C) - 1


def optim(feat, label, feat_val, label_val, opts):
    return jMultiVerseOptimizer(feat, label, feat_val, label_val, opts)