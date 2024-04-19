"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2019]-"Henry gas solubility optimization: A novel physics-based algorithm"

run with : `python3 runner.py --algo "hgso"`
"""
from fitness_function import fitness_function
import numpy as np


def j_henry_gas_solubility_optimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    num_gas = 2  # number of gas types / cluster
    K = 1  # constant
    alpha = 1  # influence of other gas
    beta = 1  # constant
    L1 = 5e-3
    L2 = 100
    L3 = 1e-2
    Ttheta = 298.15
    eps = 0.05
    c1 = 0.1
    c2 = 0.2

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'Nc' in opts:
        num_gas = opts['Nc']
    if 'K' in opts:
        K = opts['K']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'beta' in opts:
        beta = opts['beta']
    if 'L1' in opts:
        L1 = opts['L1']
    if 'L2' in opts:
        L2 = opts['L2']
    if 'L3' in opts:
        L3 = opts['L3']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(X, opts):
        return fitness_function(feat, label, feat_val, label_val, X, opts)

    # Number of dimensions
    dim = feat.shape[1]

    # Number of gas in Nc cluster
    Nn = int(np.ceil(N / num_gas))

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Henry constant & E/R constant
    H = np.random.rand(num_gas) * L1
    C = np.random.rand(num_gas) * L3
    P = np.random.rand(num_gas, Nn) * L2

    # Divide the population into Nc type of gas cluster
    Cx = [X[i*Nn:(i+1)*Nn] for i in range(num_gas)]

    # Fitness of each cluster
    Cfit = []
    fitCB = np.ones(num_gas)
    Cxb = np.zeros((num_gas, dim))
    fitG = np.inf
    for j in range(num_gas):
        Cfit.append(np.zeros(Nn))
        for i in range(len(Cx[j])):
            Cfit[j][i] = fun(Cx[j][i], opts)
            # Update best gas
            if Cfit[j][i] < fitCB[j]:
                fitCB[j] = Cfit[j][i]
                Cxb[j] = Cx[j][i]
            # Update global best
            if Cfit[j][i] < fitG:
                fitG = Cfit[j][i]
                Xgb = Cx[j][i]

    # Pre
    S = np.zeros((num_gas, Nn))

    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    t = 1

    # Iterations
    while t <= max_Iter:
        # Compute temperature
        T = np.exp(-t / max_Iter)
        for j in range(num_gas):
            # Update henry coefficient
            H[j] *= np.exp(-C[j] * ((1 / T) - (1 / Ttheta)))
            for i in range(len(Cx[j])):
                # Update solubility
                S[j][i] = K * H[j] * P[j][i]
                # Compute gamma
                gamma = beta * np.exp(-((fitG + eps) / (Cfit[j][i] + eps)))
                # Flag change between - & +
                F = -1 if np.random.rand() > 0.5 else 1
                for d in range(dim):
                    # Random constant
                    r = np.random.rand()
                    # Position update
                    Cx[j][i][d] += F * r * gamma * (Cxb[j][d] - Cx[j][i][d]) + F * r * alpha * (
                                S[j][i] * Xgb[d] - Cx[j][i][d])
                # Boundary
                Cx[j][i] = np.clip(Cx[j][i], lb, ub)

        # Fitness
        for j in range(num_gas):
            for i in range(len(Cx[j])):
                # Fitness
                Cfit[j][i] = fun(Cx[j][i], opts)

        # Select the worst solution
        Nw = int(round(N * (np.random.rand() * (c2 - c1) + c1)))
        XX = np.concatenate(Cx)
        FF = np.concatenate(Cfit)
        idx = np.argsort(FF)[::-1]
        # Update position of worst solution
        for i in range(Nw):
            XX[idx[i]] = lb + np.random.rand(dim) * (ub - lb)
            # Fitness
            FF[idx[i]] = fun(XX[idx[i]], opts)

        # Divide the population into Nc type of gas cluster back
        start_idx = 0
        for j in range(num_gas):
            end_idx = start_idx + len(Cx[j])
            Cx[j] = XX[start_idx:end_idx]
            Cfit[j] = FF[start_idx:end_idx]
            start_idx = end_idx

        # Update best solution
        for j in range(num_gas):
            for i in range(len(Cx[j])):
                # Update best gas
                if Cfit[j][i] < fitCB[j]:
                    fitCB[j] = Cfit[j][i]
                    Cxb[j] = Cx[j][i]
                # Update global best
                if Cfit[j][i] < fitG:
                    fitG = Cfit[j][i]
                    Xgb = Cx[j][i]

        curve[t] = fitG
        print('Iteration {} Best (HGSO)= {}'.format(t, curve[t]))
        t += 1

    # Select features
    Sf = np.where(Xgb > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    HGSO = {'sf': Sf.tolist(), 'ff': sFeat.tolist(), 'nf': len(Sf), 'c': curve.tolist(), 'f': feat.tolist(),
            'l': label.tolist()}
    return HGSO

def optim(feat, label, feat_val, label_val, opts):
    return j_henry_gas_solubility_optimization(feat, label, feat_val, label_val, opts)