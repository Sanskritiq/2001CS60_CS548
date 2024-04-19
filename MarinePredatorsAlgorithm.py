"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

%[2020]-"Marine Predators Algorithm: A nature-inspired metaheuristic"

run with : `python3 runner.py --algo "mpa"`
"""

import numpy as np
from fitness_function import fitness_function

def jMarinePredatorsAlgorithm(feat, label,  feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    beta = 1.5  # levy component
    P = 0.5  # constant
    FADs = 0.2  # fish aggregating devices effect

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'thres' in opts:
        thres = opts['thres']
    if 'P' in opts:
        P = opts['P']
    if 'FADs' in opts:
        FADs = opts['FADs']

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.rand(N, dim) * (ub - lb) + lb

    fit = np.zeros(N)
    fitG = np.inf

    curve = np.inf * np.ones(max_Iter+1)
    t = 1

    while t <= max_Iter:
        # Fitness
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Memory saving
        if t == 1:
            fitM = np.copy(fit)
            Xmb = np.copy(X)

        for i in range(N):
            if fitM[i] < fit[i]:
                fit[i] = fitM[i]
                X[i, :] = Xmb[i, :]

        Xmb = np.copy(X)
        fitM = np.copy(fit)

        # Construct elite
        Xe = np.tile(Xgb, (N, 1))

        # Adaptive parameter
        CF = (1 - (t / max_Iter)) ** (2 * (t / max_Iter))

        # First phase
        if t <= max_Iter / 3:
            for i in range(N):
                RB = np.random.randn(dim)
                for d in range(dim):
                    R = np.random.rand()
                    stepsize = RB[d] * (Xe[i, d] - RB[d] * X[i, d])
                    X[i, d] += P * R * stepsize
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Second phase
        elif t > max_Iter / 3 and t <= 2 * max_Iter / 3:
            for i in range(N):
                # First half update
                if i < N // 2:
                    RL = 0.05 * jLevy(beta, dim)
                    for d in range(dim):
                        R = np.random.rand()
                        stepsize = RL[d] * (Xe[i, d] - RL[d] * X[i, d])
                        X[i, d] += P * R * stepsize
                # Another half update
                else:
                    RB = np.random.randn(dim)
                    for d in range(dim):
                        stepsize = RB[d] * (RB[d] * Xe[i, d] - X[i, d])
                        X[i, d] = Xe[i, d] + P * CF * stepsize
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Third phase
        elif t > 2 * max_Iter / 3:
            for i in range(N):
                RL = 0.05 * jLevy(beta, dim)
                for d in range(dim):
                    stepsize = RL[d] * (RL[d] * Xe[i, d] - X[i, d])
                    X[i, d] = Xe[i, d] + P * CF * stepsize
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Fitness
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Memory saving
        for i in range(N):
            if fitM[i] < fit[i]:
                fit[i] = fitM[i]
                X[i, :] = Xmb[i, :]

        Xmb = np.copy(X)
        fitM = np.copy(fit)

        # Eddy formation and FADs effect
        if np.random.rand() <= FADs:
            for i in range(N):
                # Compute U
                U = np.random.rand(dim) < FADs
                for d in range(dim):
                    R = np.random.rand()
                    X[i, d] += CF * (lb + R * (ub - lb)) * U[d]
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)
        else:
            # Uniform random number [0,1]
            r = np.random.rand()
            # Define two prey randomly
            Xr1 = X[np.random.permutation(N), :]
            Xr2 = X[np.random.permutation(N), :]
            for i in range(N):
                for d in range(dim):
                    X[i, d] += (FADs * (1 - r) + r) * (Xr1[i, d] - Xr2[i, d])
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Save
        curve[t - 1] = fitG
        print('Iteration %d Best (MPA)= %f' % (t, curve[t - 1]))
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    MPA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return MPA


# Levy distribution
def jLevy(beta, dim):
    num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / deno) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    LF = u / (np.abs(v) ** (1 / beta))
    return LF


def optim(feat, label, feat_val, label_val, opts):
    return jMarinePredatorsAlgorithm(feat, label, feat_val, label_val, opts)