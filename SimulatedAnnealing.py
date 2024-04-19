"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

run with : `python3 runner.py --algo "sa"`
"""

import numpy as np
from fitness_function import fitness_function

def jSimulatedAnnealing(feat, label, feat_val, label_val,opts):
    # Parameters
    c = 0.93  # cooling rate
    T0 = 100   # initial temperature

    if 'T' in opts:
        max_Iter = opts['T']
    if 'c' in opts:
        c = opts['c']
    if 'T0' in opts:
        T0 = opts['T0']

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = jInitialization(dim)

    # Fitness
    fit = fun(feat, label, X, opts)

    # Initial best
    Xgb = X.copy()
    fitG = fit

    # Pre
    curve = np.zeros(max_Iter + 1)
    t = 1

    # Iterations
    while t <= max_Iter:
        # Probabilty of swap, insert, flip & eliminate
        prob = np.random.randint(1, 5)

        # Swap operation
        if prob == 1:
            Xnew = X.copy()
            bit0 = np.where(X == 0)[0]
            bit1 = np.where(X == 1)[0]
            len_0 = len(bit0)
            len_1 = len(bit1)
            if len_0 != 0 and len_1 != 0:
                ind0 = np.random.randint(0, len_0)
                ind1 = np.random.randint(0, len_1)
                Xnew[bit0[ind0]] = 1
                Xnew[bit1[ind1]] = 0

        # Insert operation
        elif prob == 2:
            Xnew = X.copy()
            bit0 = np.where(X == 0)[0]
            len_0 = len(bit0)
            if len_0 != 0:
                ind = np.random.randint(0, len_0)
                Xnew[bit0[ind]] = 1

        # Eliminate operation
        elif prob == 3:
            Xnew = X.copy()
            bit1 = np.where(X == 1)[0]
            len_1 = len(bit1)
            if len_1 != 0:
                ind = np.random.randint(0, len_1)
                Xnew[bit1[ind]] = 0

        # Flip operation
        elif prob == 4:
            Xnew = 1 - X

        # Fitness
        Fnew = fun(feat, label, Xnew, opts)

        # Global best update
        if Fnew <= fitG:
            Xgb = Xnew.copy()
            fitG = Fnew
            X = Xnew.copy()
        else:
            # Delta energy
            delta = Fnew - fitG
            # Boltzmann Probility
            P = np.exp(-delta / T0)
            if np.random.rand() <= P:
                X = Xnew.copy()

        # Temperature update
        T0 = c * T0

        # Save
        curve[t - 1] = fitG
        print('Iteration', t, 'Best (SA)=', curve[t - 1])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb == 1]
    sFeat = feat[:, Sf]

    # Store results
    SA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return SA

def jInitialization(dim):
    # Initialize X vectors
    X = np.zeros(dim)
    for d in range(dim):
        if np.random.rand() > 0.5:
            X[ d] = 1
    return X

def optim(feat, label, feat_val, label_val, opts):
    # Select algorithm to run
    return jSimulatedAnnealing(feat, label, feat_val, label_val, opts)