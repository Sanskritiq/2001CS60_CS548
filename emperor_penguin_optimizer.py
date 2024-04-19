import numpy as np
from fitness_function import fitness_function

def emperor_penguin_optimizer(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    M = 2     # movement parameter
    f = 3     # control parameter
    l = 2     # control parameter

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'M' in opts:
        M = opts['M']
    if 'f' in opts:
        f = opts['f']
    if 'l' in opts:
        l = opts['l']
    if 'thres' in opts:
        thres = opts['thres']

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Pre
    fit = np.zeros(N)
    fitG = np.inf

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 0

    # Iterations
    while t < max_Iter:
        for i in range(N):
            # Fitness
            fit[i] = fitness_function(feat, label, feat_val, label_val, (X[i, :] > thres), opts)
            # Best solution
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Generate radius in [0,1]
        R = np.random.rand()
        # Time
        if R > 1:
            T0 = 0
        else:
            T0 = 1
        # Temperature profile
        T = T0 - (max_Iter / (t - max_Iter))
        for i in range(N):
            for d in range(dim):
                # Pgrid
                P_grid = np.abs(Xgb[d] - X[i, d])
                # Vector A
                A = (M * (T + P_grid) * np.random.rand()) - T
                # Vector C
                C = np.random.rand()
                # Compute function S
                S = np.sqrt(f * np.exp(t / l) - np.exp(-t)) ** 2
                # Distance
                Dep = np.abs(S * Xgb[d] - C * X[i, d])
                # Position update
                X[i, d] = Xgb[d] - A * Dep
                # Boundary
                X[i, d] = np.clip(X[i, d], lb, ub)

        curve[t] = fitG
        print('Iteration', t, 'Best (EPO)=', curve[t])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    EPO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return EPO

def optim(feat, label, feat_val, label_val, opts):
    return emperor_penguin_optimizer(feat, label, feat_val, label_val, opts)
