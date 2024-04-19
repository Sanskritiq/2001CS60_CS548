"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2015]-"A human learning optimization algorithm and its application 
to multi-dimensional knapsack problems"

run with : `python3 runner.py --algo "hlo"`
"""

from fitness_function import fitness_function
import numpy as np


def jHumanLearningOptimization(feat, label, feat_val, label_val, opts):
    # Parameters
    pi = 0.85  # probability of individual learning
    pr = 0.1  # probability of exploration learning

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_Iter = opts["T"]
    if "pi" in opts:
        pi = opts["pi"]
    if "pr" in opts:
        pr = opts["pr"]

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = jInitialPopulation(N, dim)

    # Fitness
    fit = np.zeros(N)
    fitSKD = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :], opts)
        # Update SKD/gbest
        if fit[i] < fitSKD:
            fitSKD = fit[i]
            SKD = X[i, :]

    # Get IKD/pbest
    fitIKD = fit.copy()
    IKD = X.copy()

    # Pre
    curve = np.zeros(max_Iter + 1)
    curve[0] = fitSKD
    t = 1

    # Generations
    while t < max_Iter:
        for i in range(N):
            # Update solution
            for d in range(dim):
                # Random probability in [0,1]
                r = np.random.rand()
                if 0 <= r < pr:
                    # Random exploration learning operator
                    if np.random.rand() < 0.5:
                        X[i, d] = 0
                    else:
                        X[i, d] = 1
                elif pr <= r < pi:
                    X[i, d] = IKD[i, d]
                else:
                    X[i, d] = SKD[d]

        # Fitness
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, X[i, :], opts)
            # Update IKD/pbest
            if fit[i] < fitIKD[i]:
                fitIKD[i] = fit[i]
                IKD[i, :] = X[i, :]
            # Update SKD/gbest
            if fitIKD[i] < fitSKD:
                fitSKD = fitIKD[i]
                SKD = IKD[i, :]

        curve[t] = fitSKD
        print("Generation %d Best (HLO)= %f" % (t + 1, curve[t]))
        t += 1

    # Select features based on selected index
    Sf = np.where(SKD == 1)[0]
    sFeat = feat[:, Sf]

    # Store results
    HLO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return HLO


# Binary initialization strategy
def jInitialPopulation(N, dim):
    X = np.zeros((N, dim), dtype=int)
    for i in range(N):
        for d in range(dim):
            if np.random.rand() > 0.5:
                X[i, d] = 1
    return X


def optim(feat, label, feat_val, label_val, opts):
    return jHumanLearningOptimization(feat, label, feat_val, label_val, opts)
