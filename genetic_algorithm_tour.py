import numpy as np
from fitness_function import fitness_function

# Initialization function
def jInitialization(N, dim):
    X = np.zeros((N, dim), dtype=int)
    for i in range(N):
        for d in range(dim):
            if np.random.rand() > 0.5:
                X[i, d] = 1
    return X

# Tournament Selection function
def jTournamentSelection(fit, Tour_size, N):
    Tour_idx = np.random.choice(N, Tour_size, replace=False)
    Tour_fit = fit[Tour_idx]
    idx = np.argmin(Tour_fit)
    return Tour_idx[idx]

# Genetic Algorithm function
def jGeneticAlgorithmTour(feat, label, feat_val, label_val, opts):
    # Parameters
    CR = 0.8  # crossover rate
    MR = 0.01  # mutation rate
    Tour_size = 3  # tournament size

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'CR' in opts:
        CR = opts['CR']
    if 'MR' in opts:
        MR = opts['MR']
    if 'Ts' in opts:
        Tour_size = opts['Ts']

    # Objective function
    def fun(ind):
        return fitness_function(feat, label, feat_val, label_val, ind, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Initialization
    X = jInitialization(N, dim)

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(X[i, :])
        # Best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1

    # Generations
    while t < max_Iter:
        # Preparation
        Xc1 = np.zeros((N, dim))
        Xc2 = np.zeros((N, dim))
        fitC1 = np.ones(N)
        fitC2 = np.ones(N)
        z = 0
        for i in range(N):
            if np.random.rand() < CR:
                # Select two parents
                k1 = jTournamentSelection(fit, Tour_size, N)
                k2 = jTournamentSelection(fit, Tour_size, N)
                # Store parents
                P1 = X[k1, :]
                P2 = X[k2, :]
                # Single point crossover
                ind = np.random.randint(0, dim-1)
                # Crossover between two parents
                Xc1[z, :] = np.concatenate((P1[:ind], P2[ind:]))
                Xc2[z, :] = np.concatenate((P2[:ind], P1[ind:]))
                # Mutation
                for d in range(dim):
                    # First child
                    if np.random.rand() < MR:
                        Xc1[z, d] = 1 - Xc1[z, d]
                    # Second child
                    if np.random.rand() < MR:
                        Xc2[z, d] = 1 - Xc2[z, d]
                # Fitness
                fitC1[z] = fun(Xc1[z, :])
                fitC2[z] = fun(Xc2[z, :])
                z += 1
        # Merge population
        XX = np.vstack((X, Xc1[:z, :], Xc2[:z, :]))
        FF = np.hstack((fit, fitC1[:z], fitC2[:z]))
        # Select N best solution
        idx = np.argsort(FF)
        X = XX[idx[:N], :]
        fit = FF[:N]
        # Best agent
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0, :]
        # Save
        curve[t] = fitG
        print('\nGeneration {} Best (GA Tournament)= {}'.format(t + 1, curve[t]))
        t += 1

    # Select features based on selected index
    Sf = np.where(Xgb == 1)[0]
    sFeat = feat[:, Sf]

    # Store results
    GA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}
    return GA

def optim(feat, label, feat_val, label_val, opts):
    return jGeneticAlgorithmTour(feat, label, feat_val, label_val, opts)
