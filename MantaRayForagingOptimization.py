"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2020]-"Manta ray foraging optimization: An effective bio-inspired
optimizer for engineering applications"

run with : `python3 runner.py --algo "mrfo"`
"""

from fitness_function import fitness_function
import numpy as np

def jMantaRayForagingOptimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    S = 2  # somersault factor

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'S' in opts:
        S = opts['S']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(X, opts):
        return fitness_function(feat, label, feat_val, label_val, X, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(X[i], opts)
        # Best solution
        if fit[i] < fitG:
            fitG = fit[i]
            Xbest = X[i]

    # Pre
    Xnew = np.zeros((N, dim))

    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    t = 1

    # Iteration
    while t < max_Iter:
        for i in range(N):
            # [Cyclone foraging]
            if np.random.rand() < 0.5:
                if t / max_Iter < np.random.rand():
                    # Compute beta
                    r1 = np.random.rand()
                    beta = 2 * np.exp(r1 * ((max_Iter - t + 1) / max_Iter)) * (np.sin(2 * np.pi * r1))
                    for d in range(dim):
                        # Create random solution
                        Xrand = lb + np.random.rand() * (ub - lb)
                        # First manta ray follow best food
                        if i == 0:
                            Xnew[i, d] = Xrand + np.random.rand() * (Xrand - X[i, d]) + beta * (Xrand - X[i, d])
                        # Followers follow the front manta ray
                        else:
                            Xnew[i, d] = Xrand + np.random.rand() * (X[i - 1, d] - X[i, d]) + beta * (
                                        Xrand - X[i, d])
                else:
                    # Compute beta
                    r1 = np.random.rand()
                    beta = 2 * np.exp(r1 * ((max_Iter - t + 1) / max_Iter)) * (np.sin(2 * np.pi * r1))
                    for d in range(dim):
                        # First manta ray follow best food
                        if i == 0:
                            Xnew[i, d] = Xbest[d] + np.random.rand() * (Xbest[d] - X[i, d]) + beta * (
                                        Xbest[d] - X[i, d])
                        # Followers follow the front manta ray
                        else:
                            Xnew[i, d] = Xbest[d] + np.random.rand() * (X[i - 1, d] - X[i, d]) + beta * (
                                        Xbest[d] - X[i, d])
            # [Chain foraging]
            else:
                for d in range(dim):
                    # Compute alpha
                    r = np.random.rand()
                    alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                    # First manta ray follow best food
                    if i == 0:
                        Xnew[i, d] = X[i, d] + np.random.rand() * (Xbest[d] - X[i, d]) + alpha * (
                                    Xbest[d] - X[i, d])
                    # Followers follow the front manta ray
                    else:
                        Xnew[i, d] = X[i, d] + np.random.rand() * (X[i - 1, d] - X[i, d]) + alpha * (
                                    Xbest[d] - X[i, d])
            # Boundary
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Fitness
        for i in range(N):
            Fnew = fun(Xnew[i], opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i] = Xnew[i]
            # Update best
            if fit[i] < fitG:
                fitG = fit[i]
                Xbest = X[i]

        # [Somersault foraging]
        for i in range(N):
            # Manta ray update
            r2 = np.random.rand()
            r3 = np.random.rand()
            for d in range(dim):
                Xnew[i, d] = X[i, d] + S * (r2 * Xbest[d] - r3 * X[i, d])
            # Boundary
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Fitness
        for i in range(N):
            Fnew = fun(Xnew[i], opts)
            # Greedy selection
            if Fnew < fit[i]:
                fit[i] = Fnew
                X[i] = Xnew[i]
            # Update best
            if fit[i] < fitG:
                fitG = fit[i]
                Xbest = X[i]

        curve[t] = fitG
        print('Iteration {} Best (MRFO)= {}'.format(t, curve[t]))
        t += 1

    # Select features based on selected index
    Sf = np.where(Xbest > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    MRFO = {'sf': Sf.tolist(), 'ff': sFeat.tolist(), 'nf': len(Sf), 'c': curve.tolist(), 'f': feat.tolist(),
            'l': label.tolist()}
    return MRFO


def optim(feat, label, feat_val, label_val, opts):
    return jMantaRayForagingOptimization(feat, label, feat_val, label_val, opts)