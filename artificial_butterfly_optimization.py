import numpy as np
from fitness_function import fitness_function

def artificial_butterfly_optimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    step_e = 0.05   # control number of sunspot 
    ratio = 0.2     # control step
    type = 1        # type 1 or 2

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'ratio' in opts:
        ratio = opts['ratio']
    if 'stepe' in opts:
        step_e = opts['stepe']
    if 'ty' in opts:
        type = opts['ty']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    fun = fitness_function

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, feat_val, label_val, (X[i, :] > thres), opts)
        # Global update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    Xnew = np.zeros((N, dim))

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1

    # Iteration
    while t <= max_Iter:
        # Sort butterfly
        idx = np.argsort(fit)
        fit = fit[idx]
        X = X[idx]

        # Proportion of sunspot butterfly decreasing from 0.9 to ratio
        num_sun = round(N * (0.9 - (0.9 - ratio) * (t / max_Iter)))

        # Define a, linearly decrease from 2 to 0
        a = 2 - 2 * (t / max_Iter)

        # Step update
        step = 1 - (1 - step_e) * (t / max_Iter)

        # {1} Some butterflies with better fitness: Sunspot butterfly
        for i in range(num_sun):
            # Random select a butterfly k, but not equal to i
            k = np.random.choice([x for x in range(N) if x != i])
            if type == 1:
                # Randomly select a dimension
                J = np.random.randint(dim)
                # Random number in [-1,1]
                r1 = -1 + 2 * np.random.rand()
                # Position update
                Xnew[i, :] = X[i, :]
                Xnew[i, J] = X[i, J] + (X[i, J] - X[k, J]) * r1
            elif type == 2:
                # Distance
                dist = np.linalg.norm(X[k, :] - X[i, :])
                r2 = np.random.rand()
                for d in range(dim):
                    # Position update
                    Xnew[i, d] = X[i, d] + ((X[k, d] - X[i, d]) / dist) * (ub - lb) * step * r2
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

        # Fitness
        for i in range(num_sun):
            # Fitness
            Fnew = fun(feat, label, feat_val, label_val, (Xnew[i, :] > thres), opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i, :] = Xnew[i, :]
            # Global update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # {2} Some butterflies: Canopy butterfly
        for i in range(num_sun, N):
            # Random select a sunspot butterfly
            k = np.random.randint(num_sun)
            if type == 1:
                # Randomly select a dimension
                J = np.random.randint(dim)
                # Random number in [-1,1]
                r1 = -1 + 2 * np.random.rand()
                # Position update
                Xnew[i, :] = X[i, :]
                Xnew[i, J] = X[i, J] + (X[i, J] - X[k, J]) * r1
            elif type == 2:
                # Distance
                dist = np.linalg.norm(X[k, :] - X[i, :])
                r2 = np.random.rand()
                for d in range(dim):
                    # Position update
                    Xnew[i, d] = X[i, d] + ((X[k, d] - X[i, d]) / dist) * (ub - lb) * step * r2
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

        # Fitness
        for i in range(num_sun, N):
            # Fitness
            Fnew = fun(feat, label, feat_val, label_val, (Xnew[i, :] > thres), opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i, :] = Xnew[i, :]
            else:
                # Random select a butterfly
                k = np.random.randint(N)
                # Fly to new location
                r3 = np.random.rand()
                r4 = np.random.rand()
                for d in range(dim):
                    # Compute D
                    Dx = np.abs(2 * r3 * X[k, d] - X[i, d])
                    # Position update
                    X[i, d] = X[k, d] - 2 * a * r4 - a * Dx
                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)
                # Fitness
                fit[i] = fun(feat, label, feat_val, label_val, (X[i, :] > thres), opts)
            # Global update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        curve[t - 1] = fitG
        if type == 1:
            print('Iteration', t, 'Best (ABO 1)=', curve[t - 1])
        elif type == 2:
            print('Iteration', t, 'Best (ABO 2)=', curve[t - 1])
        t += 1

    # Select features
    Sf = np.nonzero(Xgb > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    ABO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return ABO

def optim(feat, label, feat_val, label_val, opts):
    return artificial_butterfly_optimization(feat, label, feat_val, label_val, opts)
