"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

%[2018]-"Butterfly optimization algorithm: a novel approach for global
%optimization"

run with : `python3 runner.py --algo "boa"`
"""

from fitness_function import fitness_function
import numpy as np


import numpy as np


def j_butterfly_optimization_algorithm(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    c = 0.01  # modular modality
    p = 0.8  # switch probability
    N = 50  # population size
    max_iter = 100  # maximum number of iterations

    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "c" in opts:
        c = opts["c"]
    if "p" in opts:
        p = opts["p"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    def fun(
        X,
    ):
        return fitness_function(feat, label, feat_val, label_val, X, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial population
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Pre-allocate
    X_new = np.zeros_like(X)
    fitG = np.inf
    curve = np.zeros(max_iter)

    # Iterations
    for t in range(max_iter):
        # Fitness evaluation
        fit = np.array([fun(x) for x in X])

        # Global update
        min_index = np.argmin(fit)
        if fit[min_index] < fitG:
            fitG = fit[min_index]
            Xgb = X[min_index].copy()

        # Power component
        a = 0.1 + 0.2 * (t / max_iter)

        for i in range(N):
            # Compute fragrance
            f = c * (fit[i] ** a)

            # Random number in [0, 1]
            r = np.random.rand()

            if r < p:
                r1 = np.random.rand()
                # Move toward the best butterfly
                X_new[i] = X[i] + ((r1**2) * Xgb - X[i]) * f
            else:
                # Randomly select two butterflies
                J, K = np.random.choice(N, 2, replace=False)
                r2 = np.random.rand()
                # Move randomly
                X_new[i] = X[i] + ((r2**2) * X[J] - X[K]) * f

            # Boundary handling
            X_new[i] = np.clip(X_new[i], lb, ub)

        # Replace population
        X = X_new.copy()

        # Save best fitness value
        curve[t] = fitG
        print("Iteration {} Best (BOA) = {}".format(t + 1, curve[t]))

    # Select features
    Sf = np.where(Xgb > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    BOA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return BOA


def optim(feat, label, feat_val, label_val, opts):
    return j_butterfly_optimization_algorithm(feat, label, feat_val, label_val, opts)
