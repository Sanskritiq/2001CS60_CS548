

'''2001CS57
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2017]-"Weighted Superposition Attraction (WSA): A swarm 
%intelligence algorithm for optimization problems – Part 1: 
%Unconstrained optimization

https://pdf.sciencedirectassets.com/272229/1-s2.0-S1568494617X00057/1-s2.0-S1568494615006766/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIB1zmBILAvmlWkVhKLqyDE7KwFbSAeTbjZhQ1lVPKcUyAiEAiMDWF6Yo9PEzMMb9OCSwqs4C3TZLx4k87CgTAmL7bL8qvAUI%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDIJLnpzlE3plrwgkNyqQBSiN0RfHy9ivszVocVG9uo8Hb9pm96XOhVdadSRN790yyzWyBSyzLU8dMoTubYQhyJhmv0P3F1EmLhi7qKiO2VNpDMxXPUpZAwq7105iN%2F3jHLkWxmOEGCwsPmnRMszTEv8XmeUS%2BHwp5b%2B766pdVV59lv%2F365VZV2lS1OLGvhvEqRxr14%2B9MCznoPp2%2FKc2ZksZb1h69XvyKBNfDsREqAA3EWemDbtc8fRRhVRbYiwC%2FTZfn%2Fhuvk0lkgYxIsj6mW%2B3r9TBA%2F4FoBY9ScbhadFHTlNEeOaGV8ocZr8tSE14St720yoa7Yd6RuYHUL1ZDkOjwOqH8Fcpeo3duoSddxcjVYN3pjxWECLj2YRncKKHDbqAEX%2BW%2FTnSJWe5VyH4ff0t8rkjNzRNbb8RCNMkjKf2xzYpTDG4WP69xLjvLvI3AEcALnHU1cYBmZLb%2FGz9IP41CCmjRe9NTlOHIsFQgEja1gNyotcukQtbmyU0r2fNQix1CWxg3GsC%2BlPiB%2B86xv%2FQxX7XM6p6KLPgg8KsAJ7nSj%2BnkCfYuTG7SVj19F4l0%2FQVBgLSwYZ%2FzK2bjRatKavod4vMAcNnNnwfAZiqlsNuWzf8ZnufhhB%2FFJVJ%2FvAX47w8fl20J%2B%2FHpGXz2%2BCXL1Puha4NAXRrkAxCIRHLLuTuOBbpTjU2AgcxtCLu0PH%2FwDYeJllc6VL%2B2Rh5fT0DyAg%2FKpONoewh4fYKzQZhnlCJjFK04zKbUpjv63SqZvxq0OrRFXU0BwKlpOBbKgBidBTl1s%2BaMMLJCVtrbfhsZESw5aRP5zP2xBpAAO21TP%2F21oa7Erff1G9%2BH8Hm3HVYflxpNc0ZrvZ7Q158LF2fhGwDG6ypZQ4yR9XlwSyXqFdiMI6JiLEGOrEBfeAy5xGGDjzaWsBK3%2Fe0t7Jk1xNrKucE4xGrK6lSGBH2vbf0HnhYsbFG4UjbMOZ47RTq8tcKFz9xGpBLP0uNWkG5R0lzIM5XPP1JPNe7QC5hKJfCi6aldveA3R6uDYMLoESdN4C8x%2BEXKUhXRzb3m9nXMVWujaf0Fu4%2Bn%2FA0%2BesuCPZ3SlhKt4TbWhhf6OR95oxPQw43cx5IN6hXuCHGoWMS283jsgEmeP62RujtAJMf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T063644Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSYAXMZHX%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=15f579ff53fb2d993ee287af96d8895fa7580e2edd1628ff407552e72324133b&hash=6f485d67a430af0934e39ee4ee993b2fcb59ff068a0b0ffcf51c524dff018995&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1568494615006766&tid=spdf-128e5cc5-fe98-43c5-91af-b1b937373194&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565903075e&rr=876ae1cacba133ae&cc=in

run with : `python3 runner.py --algo "wsa"`
'''
import numpy as np
from fitness_function import fitness_function


def jWeightedSuperpositionAttraction(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    tau = 0.8  # constant
    sl = 0.035  # step length
    phi = 0.001  # constant
    lambda_val = 0.75  # constant

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'tau' in opts:
        tau = opts['tau']
    if 'sl' in opts:
        sl = opts['sl']
    if 'phi' in opts:
        phi = opts['phi']
    if 'lambda' in opts:
        lambda_val = opts['lambda']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(feats, labels, threshold, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, threshold, options)

    # Number of dimensions
    dim = feat.shape[1]
    
    # Initial
    X = np.random.uniform(lb, ub, (N, dim))

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    # Iterations
    while t <= max_Iter:
        # Rank solution based on fitness
        idx = np.argsort(fit)
        X = X[idx]

        # {1} Target point determination: Figure 2
        w = np.zeros(N)
        Xtar = np.zeros(dim)
        for i in range(1,N):
            # Assign weight based on rank
            w[i-1] = i ** (-1 * tau)
            # Create target
            for d in range(dim):
                Xtar[d] += X[i, d] * w[i]

        # Boundary
        Xtar[Xtar > ub] = ub
        Xtar[Xtar < lb] = lb

        # Fitness
        fitT = fun(feat, label, (Xtar > thres), opts)

        # Best update
        if fitT < fitG:
            fitG = fitT
            Xgb = Xtar

        # {2} Compute search direction: Figure 4
        gap = np.zeros((N, dim))
        direct = np.zeros((N, dim))
        for i in range(N):
            if fit[i] >= fitT:
                for d in range(dim):
                    # Compute gap
                    gap[i, d] = Xtar[d] - X[i, d]
                    # Compute direction
                    direct[i, d] = np.sign(gap[i, d])
            elif fit[i] < fitT:
                if np.random.rand() < np.exp(fit[i] - fitT):
                    for d in range(dim):
                        # Compute gap
                        gap[i, d] = Xtar[d] - X[i, d]
                        # Compute direction
                        direct[i, d] = np.sign(gap[i, d])
                else:
                    for d in range(dim):
                        # Compute direction
                        direct[i, d] = np.sign(-1 + (1 + 1) * np.random.rand())

        # Compute step sizing function (2)
        if np.random.rand() <= lambda_val:
            sl = sl - np.exp(t / (t - 1)) * phi * sl
        else:
            sl = sl + np.exp(t / (t - 1)) * phi * sl

        # {3} Neighbor generation: Figure 7
        for i in range(N):
            for d in range(dim):
                # Update (1)
                X[i, d] = X[i, d] + sl * direct[i, d] * np.abs(X[i, d])
            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        # Fitness
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        curve[t - 1] = fitG
        print('Iteration', t, 'Best (WSA)=', curve[t - 1])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    WSA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return WSA


def optim(feat, label, feat_val, label_val, opts):
    return jWeightedSuperpositionAttraction(feat, label, feat_val, label_val, opts)