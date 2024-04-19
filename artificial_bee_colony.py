import numpy as np
import time
from fitness_function import fitness_function

def optim(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    max_limit = 5  # Maximum limits allowed

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'max' in opts:
        max_limit = opts['max']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(X):
        return fitness_function(feat, label, feat_val, label_val, (X > thres), opts)

    # Number of dimensions
    dim = feat.shape[1]
    # Divide into employ and onlooker bees
    N = N // 2
    # Initial
    X = np.random.uniform(lb, ub, size=(N, dim))
    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(X[i])
        # Best food source
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i]

    # Pre
    limit = np.zeros(N)
    V = np.zeros((N, dim))

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1
    # Iteration
    while t < max_Iter:
        # {1} Employed bee phase
        for i in range(N):
            # Choose k randomly, but not equal to i
            k = np.delete(np.arange(N), i)
            k = np.random.choice(k)
            for d in range(dim):
                # Phi in [-1,1]
                phi = -1 + 2 * np.random.rand()
                # Position update (6)
                V[i, d] = X[i, d] + phi * (X[i, d] - X[k, d])
                # Boundary
                V[i, d] = min(max(V[i, d], lb), ub)

        # Fitness
        for i in range(N):
            # Fitness
            Fnew = fun(V[i])
            # Compare neighbor bee
            if Fnew <= fit[i]:
                # Update bee & reset limit counter
                X[i] = V[i]
                fit[i] = Fnew
                limit[i] = 0
            else:
                # Update limit counter
                limit[i] += 1

        # Minimization problem (5)
        Ifit = 1 / (1 + fit)
        # Convert probability (7)
        prob = Ifit / np.sum(Ifit)

        # {2} Onlooker bee phase
        i = 0
        m = 0
        while m < N:
            if np.random.rand() < prob[i]:
                # Choose k randomly, but not equal to i
                k = np.delete(np.arange(N), i)
                k = np.random.choice(k)
                for d in range(dim):
                    # Phi in [-1,1]
                    phi = -1 + 2 * np.random.rand()
                    # Position update (6)
                    V[i, d] = X[i, d] + phi * (X[i, d] - X[k, d])
                    # Boundary
                    V[i, d] = min(max(V[i, d], lb), ub)
                # Fitness
                Fnew = fun(V[i])
                # Greedy selection
                if Fnew <= fit[i]:
                    X[i] = V[i]
                    fit[i] = Fnew
                    limit[i] = 0
                    # Re-compute new probability (5,7)
                    Ifit = 1 / (1 + fit)
                    prob = Ifit / np.sum(Ifit)
                else:
                    limit[i] += 1
                m += 1

            # Reset i
            i += 1
            if i >= N:
                i = 0

        # {3} Scout bee phase
        for i in range(N):
            if limit[i] >= max_limit:
                # Produce new bee (8)
                X[i] = np.random.uniform(lb, ub, dim)
                # Reset Limit
                limit[i] = 0
                # Fitness
                fit[i] = fun(X[i])
            # Best food source
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i]

        curve[t] = fitG
        print('Iteration %d Best (ABC)= %f' % (t, curve[t]))
        t += 1

    # Select features based on selected index
    Sf = np.where(Xgb > thres)[0]
    sFeat = feat[:, Sf]
    # Store results
    ABC = {
        'sf': Sf.tolist(),
        'ff': sFeat.tolist(),
        'nf': len(Sf),
        'c': curve.tolist(),
        'f': feat.tolist(),
        'l': label.tolist()
    }
    return ABC
