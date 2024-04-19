"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

%[2012]-"A new fruit fly optimization algorithm: Taking the financial 
%distress model as an example"  

run with : `python3 runner.py --algo "foa"`
"""

from fitness_function import fitness_function
import numpy as np

import numpy as np


def j_fruit_fly_optimization_algorithm(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_Iter = opts["T"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    def fun(X, opts):
        return fitness_function(feat, label, feat_val, label_val, X, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))
    Y = np.random.uniform(lb, ub, (N, dim))

    # Compute solution
    S = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            # Distance between X and Y axis
            dist = np.sqrt(X[i, d] ** 2 + Y[i, d] ** 2)
            # Solution
            S[i, d] = 1 / dist
            # Boundary
            S[i, d] = np.clip(S[i, d], lb, ub)

    # Pre
    fit = np.zeros(N)
    fitG = np.inf
    curve = np.inf * np.ones(max_Iter+1)
    t = 1

    # Iterations
    while t <= max_Iter:
        # Fitness
        for i in range(N):
            # Fitness
            fit[i] = fun(S[i], opts)
            # Update better solution
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = S[i]
                # Update X & Y
                Xb = X[i]
                Yb = Y[i]

        for i in range(N):
            for d in range(dim):
                # Random in [-1,1]
                r1 = -1 + 2 * np.random.rand()
                r2 = -1 + 2 * np.random.rand()
                # Compute new X & Y
                X[i, d] = Xb[d] + (ub - lb) * r1
                Y[i, d] = Yb[d] + (ub - lb) * r2
                # Distance between X and Y axis
                dist = np.sqrt(X[i, d] ** 2 + Y[i, d] ** 2)
                # Solution
                S[i, d] = 1 / dist
                # Boundary
                S[i, d] = np.clip(S[i, d], lb, ub)

        curve[t] = fitG
        print("Generation {} Best (FOA)= {}".format(t, curve[t]))
        t += 1

    # Select features
    Sf = np.where(Xgb > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    FOA = {
        "sf": Sf.tolist(),
        "ff": sFeat.tolist(),
        "nf": len(Sf),
        "c": curve.tolist(),
        "f": feat.tolist(),
        "l": label.tolist(),
    }
    return FOA


def optim(feat, label, feat_val, label_val, opts):
    return j_fruit_fly_optimization_algorithm(feat, label, feat_val, label_val, opts)