import numpy as np
from fitness_function import fitness_function

def crow_search_algorithm(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    AP = 0.1   # awareness probability
    fl = 1.5   # flight length

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'AP' in opts:
        AP = opts['AP']
    if 'fl' in opts:
        fl = opts['fl']
    if 'thres' in opts:
        thres = opts['thres']
    
    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fitness_function(feat, label, feat_val, label_val, (X[i, :] > thres), opts)
        # Global update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Save memory
    fitM = fit.copy()
    Xm = X.copy()

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 0

    # Iteration
    while t < max_Iter:
        for i in range(N):
            # Random select 1 memory crow to follow
            k = np.random.randint(0, N)
            # Awareness of crow m
            if np.random.rand() >= AP:
                r = np.random.rand()
                for d in range(dim):
                    # Crow m does not know it has been followed
                    X[i, d] = X[i, d] + r * fl * (Xm[k, d] - X[i, d])
            else:
                for d in range(dim):
                    # Crow m fools crow i by flying randomly
                    X[i, d] = np.random.uniform(lb, ub)
        # Fitness
        for i in range(N):
            # Fitness
            Fnew = fitness_function(feat, label, feat_val, label_val, (X[i, :] > thres), opts)
            # Check feasibility
            if np.all(X[i, :] >= lb) and np.all(X[i, :] <= ub):
                # Update crow
                fit[i] = Fnew
                # Memory update
                if fit[i] < fitM[i]:
                    Xm[i, :] = X[i, :]
                    fitM[i] = fit[i]
                    # Global update
                    if fitM[i] < fitG:
                        fitG = fitM[i]
                        Xgb = Xm[i, :]
        curve[t] = fitG
        print('Iteration', t + 1, 'Best (CSA)=', curve[t])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    CSA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return CSA


def optim(data, labels, data_val, labels_val, opts):
    return crow_search_algorithm(data, labels, data_val, labels_val, opts)