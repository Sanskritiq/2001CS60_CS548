"""2001CS57
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2018]-"Tree growth algorithm (TGA): A novel approach for solving
optimization problems"

https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197618X00049/1-s2.0-S0952197618301003/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCICUjahdQt6sL8SL2W%2B23mnk0gMBaQfZzTrWnnNehnopOAiBwXJA9ykp21GaFTEEdr0OACf80HVXmGLRPmsgQ8Vj%2FWiqzBQgQEAUaDDA1OTAwMzU0Njg2NSIMg13%2FTXuLeknAf6p4KpAFzgr2oQA7kQJK%2FDv3fMTUwg0KHWgZ42sQh2ffaaBps0qm2D1pxUS%2BUWCjOs0xtJMTGwF88JjHN5FE3xL1sPGsiemkw6b1i3p2CggIT7YmUMMVjbRjNglKU1s68ImgoquzrYTHQZn9pBhafjH5ZB4lcVR2f5%2FJtSdTVMvjsDDGoG7CiAnZmf4Zb754OuoxBpqvRz7wfyCScTcSRBrHbVRLntneq8olhy2dtSczSFD1LuerYJJpqym%2B3RdL3ZMeIkKcN8d9%2B0dXIO%2BkemtTWyqohlXFTneAc8cJLO4ELtvcJLyfVKLC66cMaptT5IamCjXal0W157xUCYjSBEtsgGmvO8q3KN6npyF2qNgS%2BIOgxpmejFDCl3rQYpuzsZ4tb6yhKwbqFPZVXp1YjY%2Babmo86wcqO8C4K027OtEM3DoQ4x6q7YGNfPP7fO7UWh%2Ff61U5Yr41tcUORXnFEPf6ApO%2BR%2Bot6Z0eY5fmpHQ6CCB0g6KEQlkUdkiUSb7HvQbIIYOkChxO7Yf%2BNoecubJkr8KZeY6YWzW5eHL5%2F%2Bj5BhZv6ya%2Bm8qZIyw0DTm8zRMu6kKJqefuehup42P8giAwBsyYA54BWkIKQeEm%2B6FNYdDbfRHC2OcnaiTCeD3ha1NjjemLkEIsrNkm7XCI3kAijS4SH6hL29vrW%2FmhaIZS6PxRKT%2Bohh%2Fe3HPkCp1PorAUfpBqBeVI0ci0Itj8U8fohDBdQNyrKajnN8Qqbpc4idBIy5n4vGBS%2BAJ%2FLkYirHWbTSGh%2F%2FMSITCBOuYP5a%2B3T5IQNqWv95n33XJbp5ChyP79Q8K2ERzyEDH1olic%2BqPES5pywKlueWtpuQ7%2Fujv%2FtLbSkrP8iLFzc4%2F4Q93MCuU6inAwlqaIsQY6sgGol8ErBpP46hb%2FmM7ONDxZ%2BlcSNqx7bVdgk%2FvlSNQavKuBP9M2FrAnY1CnttYqPYskPlB7O6KH1AHo66gNVsae5Kd9se%2FzgAAWTzgMUMPplmQ4YgwiSjVnNTgm0bChneCwlIovrZo80npV5K6QhwlPdfZuzZIYKOXkMHApUu9mbd8kc4P3sED%2BGsqVkkB2exyEk5o8zeHoKS76ZDJUgTUIV8WuXkCAJmALYR%2FmpjBoJLUq&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T074402Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRW7VCUNN%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cb1b4c552d90a738e6873b42c7244dbeb0d0fcf249eeabb1397def930a972584&hash=5ee1828b103f1a2347c6d52439cb1733f3377db008a5483c44762b954356dc6d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197618301003&tid=spdf-77bd7f4e-ef3b-46ee-a89f-6b0443871859&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565900565b&rr=876b445fecd233a8&cc=in

run with : `python3 runner.py --algo "tga"`
"""

import numpy as np
from fitness_function import fitness_function


def jTreeGrowthAlgorithm(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    num_tree1 = 3  # size of first group
    num_tree2 = 5  # size of second group
    num_tree4 = 3  # size of fourth group
    theta = 0.8  # tree reduction rate of power
    lambda_val = 0.5  # control nearest tree

    if "T" in opts:
        max_Iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "N1" in opts:
        num_tree1 = opts["N1"]
    if "N2" in opts:
        num_tree2 = opts["N2"]
    if "N4" in opts:
        num_tree4 = opts["N4"]
    if "theta" in opts:
        theta = opts["theta"]
    if "lambda" in opts:
        lambda_val = opts["lambda"]
    if "thres" in opts:
        thres = opts["thres"]

    # Limit number of N4 to N1
    if num_tree4 > num_tree1 + num_tree2:
        num_tree4 = num_tree1 + num_tree2

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
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)

    # Best solution
    fitG = np.min(fit)
    idx = np.argmin(fit)
    Xgb = X[idx, :]

    # Pre
    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    t = 1

    # Iterations
    while t <= max_Iter:
        # {1} Best trees group
        for i in range(num_tree1):
            r1 = np.random.rand()
            X1 = (X[i, :] / theta) + r1 * X[i, :]
            # Boundary
            X1 = np.clip(X1, lb, ub)
            # Fitness
            fitT = fun(feat, label, (X1 > thres), opts)
            # Greedy selection
            if fitT <= fit[i]:
                X[i, :] = X1
                fit[i] = fitT

        # {2} Competitive for light tree group
        X_ori = X.copy()
        for i in range(num_tree1, num_tree1 + num_tree2):
            # Neighbor tree
            dist = np.zeros(num_tree1 + num_tree2)
            for j in range(num_tree1 + num_tree2):
                if j != i:
                    # Compute Euclidean distance
                    dist[j] = np.sqrt(np.sum((X_ori[j, :] - X_ori[i, :]) ** 2))
                else:
                    # Solve same tree problem
                    dist[j] = np.inf
            # Find 2 trees with shorter distance
            idx = np.argsort(dist)
            T1 = X_ori[idx[0], :]
            T2 = X_ori[idx[1], :]
            # Alpha in [0,1]
            alpha = np.random.rand()
            for d in range(dim):
                # Compute linear combination between 2 shorter tree
                y = lambda_val * T1[d] + (1 - lambda_val) * T2[d]
                # Move tree i between 2 adjacent trees
                X[i, d] = X[i, d] + alpha * y
            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)
            # Fitness
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)

        # {3} Remove and replace group
        for i in range(num_tree1 + num_tree2, N):
            X[i, :] = np.random.uniform(lb, ub, dim)
            # Fitness
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)

        # {4} Reproduction group
        Xnew = np.zeros((num_tree4, dim))
        Fnew = np.zeros(num_tree4)
        for i in range(num_tree4):
            # Random a best tree
            r = np.random.randint(0, num_tree1)
            Xbest = X[r, :]
            # Mask operator
            mask = np.random.randint(0, 2, dim)
            # Mask operation between new & best trees
            for d in range(dim):
                # Generate new solution
                Xn = np.random.uniform(lb, ub)
                if mask[d] == 1:
                    Xnew[i, d] = Xbest[d]
                elif mask[d] == 0:
                    # Generate new tree
                    Xnew[i, d] = Xn
            # Fitness
            Fnew[i] = fun(feat, label, (Xnew[i, :] > thres), opts)

        # Sort population get best nPop trees
        XX = np.vstack((X, Xnew))
        FF = np.concatenate((fit, Fnew))
        idx = np.argsort(FF)
        X = XX[idx[:N], :]
        fit = FF[:N]

        # Global best
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0, :]

        curve[t] = fitG
        print("Iteration", t, "Best (TGA)=", curve[t])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    TGA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return TGA

def optim(feat, label, feat_val, label_val, opts):
    return jTreeGrowthAlgorithm(feat, label, feat_val, label_val, opts)