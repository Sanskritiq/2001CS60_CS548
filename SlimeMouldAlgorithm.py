"""2001CS57
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2020]-"Slime mould algorithm: A new method for stochastic optimization"

https://pdf.sciencedirectassets.com/271521/1-s2.0-S0167739X20X00070/1-s2.0-S0167739X19320941/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD9gZ0kEy2ehpMhUuX3bDQMu2CZbMZNFDytUsAD13XxjgIga4IBvxAEIIPGyj1JBQWBUBl7q0i1H3ny84yiqRVn22wqsgUIEBAFGgwwNTkwMDM1NDY4NjUiDGMfagA6gjSSBivpyyqPBfHX2cMDX5cf1guEnd8sfSdgSCH63X0QOGb47wtQc0rSjPEJ1nY1FFkEUljueY31N3TskX8giAu1R0sxyv2mPYwmgEGpEJIGYkvVUNb2D2RtBK6lHy9AgZ4sEf4pB7W0wvnMCpgb%2BfIE5eCTom4U6WAGYISUh0kZg8Fn9KtNc4gRQJ7Mz%2FSsvMEXRzfvOIw7njWGmIiTz%2F6rBf1kTflIgvGm1TTGclDLNNxO4fFSA3JzgfwW%2FwMggXcI999ZznsaKSpJaPGi2713M7PnPVYOy%2BwRdB7qA59ui8LxAjRrQVf%2FPsdInaYVRAgnq5%2BJQw%2BNhKTSDHUQXpVwwbbX10gpVONlGbsIaIPgwPp%2FyWuqnRBqJrK8VWT23ZC6nXg8RoIpuL%2FYj95uO90iWJmDlNM4%2FVb2aBq3CWpRZGf%2F%2BRXgH4O%2B%2BlTVHX9x2lMYTXdRvQAFhPaTE6mjDp2I16t%2BQTQX9J%2BZrin5tIfrXyUcAk%2BPSFUeVJ%2Fa33DEiim6ifsw33uYErmu61vOfyahj%2BAYpGIZBO%2FkQXBo9E8KYU%2FdWLUO53iCnhbGNQgHeFGcCkOAMcMpkEjoz40TQfsf7SB4J5OZ1jLGtzo1hETOhCLGj8BQVhQYaPAANAacxDiKOZG%2BQ%2BdLTEYnf6fcJiPCovdSeFvDE6Z34%2FDQDHYFTfTGvQu8qo6d25933erXooQStcStE9mnhVKAQDY%2FGrDRlvYKSsYV517FxbnBuPYcKcg45rtn87bfmFO4rMmWzgTJW%2FcD2SuCnEDzuiufRgND%2FIJEJPVUTcxPRpdCLfK2CP18X%2B389sQ9Dur1q%2Fern3ef07NBL49s%2Fj6s%2FSLv6khnuWAx9Jyllp5pzvxfOXRo5ZFs8vEVbJgwvrCIsQY6sQEN5Gq73pOqh%2FdSvTtiZis%2B3iM3b3VwvzQwbbFsRL0B%2BvjLNa%2BmvnnkLk5Bd4a8XUlQa%2Bti9TaW2JOXflLgij8njpnM2emSNeHFD1EIHtJDPaTcvfNJUuTUQjLsenCiJNcSh3iuCBzcPBfdKl%2Fz9tMFFKjea02oZ98ZttV29I%2FJGkk8U3mLmc0C0eqpw2BGEy9I88%2F%2FQ%2Fb%2B4TAjQHXZhf1ecRumFDVal%2BoHjjLyAwKFAt0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T081114Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUDZ3PCR7%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=44dafbd1ba54af2c4dc3168eac79612949f2b66de0d3e743a0602ffd31217cf1&hash=ece9d955f46e03a66d1bd2ca61b1c689580c84a9beec51f2a550f1340fdf3844&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167739X19320941&tid=spdf-0608aa52-f92e-44fe-86ce-7c5b95f40d6b&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565900540c&rr=876b6c348e0c3386&cc=in

run with : `python3 runner.py --algo "sma"`
"""

import numpy as np
from fitness_function import fitness_function

import numpy as np

def jSlimeMouldAlgorithm(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    z = 0.03  # control local & global

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'z' in opts:
        z = opts['z']
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

    # Pre
    fit = np.zeros(N)
    fitG = np.inf
    W = np.zeros((N, dim))

    curve = np.inf
    t = 1

    # Iteration
    while t <= max_Iter:
        # Fitness
        for i in range(N):
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best
            if fit[i] < fitG:
                fitG = fit[i]
                Xb = X[i, :]

        # Sort smell index
        idxS = np.argsort(fit)
        fitS = np.sort(fit)

        # Best fitness & worst fitness
        bF = np.min(fit)
        wF = np.max(fit)

        # Compute W
        for i in range(N):
            for d in range(dim):
                # Condition
                r = np.random.rand()
                if i <= N / 2:
                    W[idxS[i], d] = 1 + r * np.log10(((bF - fitS[i]) / (bF - wF + np.finfo(float).eps)) + 1)
                else:
                    W[idxS[i], d] = 1 - r * np.log10(((bF - fitS[i]) / (bF - wF + np.finfo(float).eps)) + 1)

        # Compute a
        a = np.arctanh(-(t / max_Iter) + 1)

        # Compute b
        b = 1 - (t / max_Iter)

        # Update
        for i in range(N):
            if np.random.rand() < z:
                X[i, :] = np.random.uniform(lb, ub, dim)
            else:
                # Update p
                p = np.tanh(np.abs(fit[i] - fitG))
                # Update vb
                vb = np.random.uniform(-a, a, dim)
                # Update vc
                vc = np.random.uniform(-b, b, dim)
                for d in range(dim):
                    # Random in [0,1]
                    r = np.random.rand()
                    # Two random individuals
                    A = np.random.randint(0, N)
                    B = np.random.randint(0, N)
                    if r < p:
                        X[i, d] = Xb[d] + vb[d] * (W[i, d] * X[A, d] - X[B, d])
                    else:
                        X[i, d] = vc[d] * X[i, d]

                # Boundary
                X[i, :] = np.clip(X[i, :], lb, ub)

        # Save
        curve = np.append(curve, fitG)
        print('Iteration', t, 'Best (SMA)=', curve[-1])
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xb > thres]
    sFeat = feat[:, Sf]

    # Store results
    SMA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return SMA

def optim(data, labels, data_val, labels_val, opts):
    return jSlimeMouldAlgorithm(data, labels, data_val, labels_val, opts)