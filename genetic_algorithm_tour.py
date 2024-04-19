import numpy as np
import random
from fitness_function import fitness_function

def genetic_algorithm_tour(feat, label, opts):
    # Parameters 
    CR = 0.8      # crossover rate
    MR = 0.01     # mutation rate
    Tour_size = 3 # tournament size

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
    def fun(X):
        return fitness_function(feat, label, X, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Initialize
    X = initialization(N, dim)
    
    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(X[i])
        # Best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i]

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    # Generations
    while t <= max_Iter:
        # Preparation  
        Xc1 = np.zeros((1, dim))
        Xc2 = np.zeros((1, dim))
        fitC1 = np.ones((1, 1))
        fitC2 = np.ones((1, 1))
        z = 0
        
        for i in range(N):
            if np.random.rand() < CR:
                # Select two parents 
                k1 = tournament_selection(fit, Tour_size, N)
                k2 = tournament_selection(fit, Tour_size, N)
                # Store parents 
                P1 = X[k1] 
                P2 = X[k2]
                # Single point crossover
                ind = np.random.randint(1, dim)
                # Crossover between two parents
                Xc1[z] = np.concatenate((P1[:ind], P2[ind:]))
                Xc2[z] = np.concatenate((P2[:ind], P1[ind:]))
                # Mutation
                for d in range(dim):
                    # First child
                    if np.random.rand() < MR:
                        Xc1[z, d] = 1 - Xc1[z, d]
                    # Second child
                    if np.random.rand() < MR:
                        Xc2[z, d] = 1 - Xc2[z, d]
                # Fitness
                fitC1[z] = fun(Xc1[z])
                fitC2[z] = fun(Xc2[z])
                z += 1
                
        # Merge population
        XX = np.vstack((X, Xc1[:z], Xc2[:z]))
        FF = np.hstack((fit, fitC1[:z].flatten(), fitC2[:z].flatten()))
        
        # Select N best solution 
        idx = np.argsort(FF)
        X = XX[idx[:N]]
        fit = FF[idx[:N]]
        
        # Best agent
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0]

        # Save
        curve[t - 1] = fitG 
        print('\nGeneration %d Best (GA Tournament)= %f' % (t, curve[t - 1]))
        t += 1
        
    # Select features based on selected index
    Sf = np.where(Xgb == 1)[0]
    sFeat = feat[:, Sf]
    
    # Store results
    GA = {
        'sf': Sf.tolist(),
        'ff': sFeat.tolist(),
        'nf': len(Sf),
        'c': curve.tolist(),
        'f': feat.tolist(),
        'l': label.tolist()
    }
    
    return GA

def tournament_selection(fit, Tour_size, N):
    # Random positions based on position & Tournament Size
    Tour_idx = np.random.choice(N, Tour_size)
    # Select fitness value based on position selected by tournament 
    Tour_fit = fit[Tour_idx]
    # Get position of best fitness value (win tournament)
    idx = np.argmin(Tour_fit)
    # Store the position
    return Tour_idx[idx]

def initialization(N, dim):
    # Initialize X vectors
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            if np.random.rand() > 0.5:
                X[i, d] = 1
    return X
