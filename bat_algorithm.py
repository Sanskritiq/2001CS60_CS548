import numpy as np
import random
from fitness_function import fitness_function

def optim(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    fmax = 2     # maximum frequency
    fmin = 0     # minimum frequency
    alpha = 0.9  # constant
    gamma = 0.9  # constant
    A_max = 2    # maximum loudness
    r0_max = 1   # maximum pulse rate

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'fmax' in opts:
        fmax = opts['fmax']
    if 'fmin' in opts:
        fmin = opts['fmin']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'gamma' in opts:
        gamma = opts['gamma']
    if 'A' in opts:
        A_max = opts['A']
    if 'r' in opts:
        r0_max = opts['r']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(X):
        return fitness_function(feat, label, feat_val, label_val, (X > thres), opts)

    # Number of dimensions
    dim = feat.shape[1]
    
    # Initialize bats
    X = np.random.uniform(lb, ub, size=(N, dim))
    V = np.zeros_like(X)
    
    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(X[i])
        # Global best
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i]

    # Loudness of each bat, [1 ~ 2]
    A = np.random.uniform(1, A_max, size=N)
    # Pulse rate of each bat, [0 ~ 1]
    r0 = np.random.uniform(0, r0_max, size=N)
    r = r0
    
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2
    
    # Iterations
    while t <= max_Iter:
        for i in range(N):
            # Beta [0~1]
            beta = random.random()
            # Frequency (2)
            freq = fmin + (fmax - fmin) * beta
            for d in range(dim):
                # Velocity update (3)
                V[i, d] = V[i, d] + (X[i, d] - Xgb[d]) * freq
                # Position update (4)
                X[i, d] = X[i, d] + V[i, d]
                
            # Generate local solution around best solution
            if random.random() > r[i]:
                for d in range(dim):
                    # Epsilon in [-1,1]
                    eps = -1 + 2 * random.random()
                    # Random walk (5)
                    X[i, d] = Xgb[d] + eps * np.mean(A)
                    
            # Boundary
            X[i] = np.clip(X[i], lb, ub)
            
        # Fitness
        for i in range(N):
            # Fitness
            Fnew = fun(X[i])
            # Greedy selection
            if random.random() < A[i] and Fnew <= fit[i]:
                fit[i] = Fnew
                # Loudness update (6)
                A[i] = alpha * A[i]
                # Pulse rate update (6)
                r[i] = r0[i] * (1 - np.exp(-gamma * t))
                
                # Global best
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i]

        curve[t - 1] = fitG
        print('\nIteration %d Best (BA)= %f' % (t, curve[t - 1]))
        t += 1
    
    # Select features
    Sf = np.where(Xgb > thres)[0]
    sFeat = feat[:, Sf]
    
    # Store results
    BA = {
        'sf': Sf.tolist(),
        'ff': sFeat.tolist(),
        'nf': len(Sf),
        'c': curve.tolist(),
        'f': feat.tolist(),
        'l': label.tolist()
    }
    
    return BA
