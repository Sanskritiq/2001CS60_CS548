"""2001CS57
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2014]-"Symbiotic organisms search: A new metaheuristic optimization algorithm" 

https://pdf.sciencedirectassets.com/271458/1-s2.0-S0045794914X00076/1-s2.0-S0045794914000881/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIE0WQBRtqYkBhlh6sU77HZm1CCLSVd6v9oW%2FMoPDhtA4AiEA8SuNUo68R%2FjWjdi4tqFNNn0XCDLdl8zLQcRCO1Eb9HIqswUIEBAFGgwwNTkwMDM1NDY4NjUiDJvhq10HMsJ20nuv7CqQBeYYdlBg%2FK0yj3PO8XAlmIH0x8J%2FtQUn%2BRSIUK7lt8nv1QeMz7t%2BOLIWaEc2qbZ9UlxXO6%2FHeMzXbJC4NN72BVbNitkYwoxMGKqlHcGHIIffFDGjCS4BiLmTBwVPvUJ3G%2BpPGXLpdwG%2BBu5UZsvJMrtXqoFjU%2BQHD1gUQIG82mIWNdGM1FPlUapHK8nVxd%2Foq9274bNEPkWtdt3efGZy2FCOz1UnWUkL6YhPDpeU3mQs5OypSp%2BDgQi7ObkFyZkQL3bPuciYnpVAVo6RB74xTczGf0Ion7XBOTn%2Fmwoxtf6SXFkKLgbiETGtA1ioMhxnoG7cNJsRUdycA3ZkKseDX8lTlChVfnxKWA13xJlKkmGEQd9p0%2Bbp6dxrhpXrXuZeFj%2BSNWpglK8rA06oXWf%2FI3djkwBt%2B8%2BodzfvAlkaWU6tnQZ50yhQWBD01e2fJBfANW%2BaNMoUMdPdfgk%2FVI%2BIHjD2510IE2fj5T%2FS3VoxcCTYP0WgtOSkmflMN%2BHw%2Frc2D4QDXtxIVtjFmrqsQmAj%2BbF5AXCsZiHmZERhsJuYIsvFfljhWezZVNMaLLSooH%2FjJL0bUTJPq5ddSQcppm6SKc%2FxwvHeq%2BmbZnIA6khx3p%2F6ONYoXO4IBZcFNqNwpGzk3F1PwLKF4u5qjZOQgty7%2FcI8GP0TnUVe09OoznqvRlo4auITV7eDns6c3MWclCKztQHrz1AV4SD%2BOidtAVPhoCmhyuEDBuas7%2F9wd3AXLtffu%2FK4K1cNoVp%2FXVNzAJvZTMuWqIgbyqBSYI9%2BvtFCt9IN9BPrQt6Xs7nbURslxRB%2Fr6CstwvrEue%2B1oXlISscL%2FlJfjqKlmeFgNiX5igELmpIvbspTErN7IPiguOT0VNxMImoiLEGOrEBEE%2BZNSyeCysZV45dkEVgxCnbPkD7EkUbie9sbdAdINaEMUe6JH7u93hXWkQRMBOK1rCOp5FqQPy7E%2B724RPMdPPjhkMJk5wnIjsZHbMXOMKAmfVTsWY0CoXfVMsVBrt3aaVQ2z992jTgceJxI49YwLRe%2BehqieUjtjn4Hd8qRWyY6wk6Prw1nR5XK4yf4yFaAFyLfdl40n1bfuzONZ0ybwGHVWFb%2Fb81S6MPb2GBq24u&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T075919Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYPETFI7P%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e3c2ff9b6020d47c9e5a16bb6af64a964a57ccbe9cd22a59198a2c31e629d01c&hash=cffedbd187d05fc60343ca4ebc2815c3519e6a62f736c3f235c9118e505ef86c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0045794914000881&tid=spdf-e1cb33aa-4069-4422-a6fa-613324d89d39&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565900570e&rr=876b5abf5f8633a2&cc=in
run with : `python3 runner.py --algo "sos"`
"""

import numpy as np
from fitness_function import fitness_function

def jSymbioticOrganismsSearch(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
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
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)

    # Best solution
    fitG = np.min(fit)
    idx = np.argmin(fit)
    Xgb = X[idx, :]

    # Pre
    Xi = np.zeros(dim)
    Xj = np.zeros(dim)

    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    t = 1

    # Iteration
    while t <= max_Iter:
        for i in range(N):
            # {1} Mutualism phase
            R = np.random.permutation(N)
            R = R[R != i]
            J = R[0]
            # Benefit factor [1 or 2]
            BF1 = np.random.randint(1, 3)
            BF2 = np.random.randint(1, 3)
            for d in range(dim):
                # Mutual vector
                MV = (X[i, d] + X[J, d]) / 2
                # Update solution
                Xi[d] = X[i, d] + np.random.rand() * (Xgb[d] - MV * BF1)
                Xj[d] = X[J, d] + np.random.rand() * (Xgb[d] - MV * BF2)
            # Boundary
            Xi = np.clip(Xi, lb, ub)
            Xj = np.clip(Xj, lb, ub)
            # Fitness
            fitI = fun(feat, label, (Xi > thres), opts)
            fitJ = fun(feat, label, (Xj > thres), opts)
            # Update if better solution
            if fitI < fit[i]:
                fit[i] = fitI
                X[i, :] = Xi
            if fitJ < fit[J]:
                fit[J] = fitJ
                X[J, :] = Xj

            # {2} Commensalism phase
            R = np.random.permutation(N)
            R = R[R != i]
            J = R[0]
            for d in range(dim):
                # Random number in [-1,1]
                r1 = -1 + 2 * np.random.rand()
                # Update solution
                Xi[d] = X[i, d] + r1 * (Xgb[d] - X[J, d])
            # Boundary
            Xi = np.clip(Xi, lb, ub)
            # Fitness
            fitI = fun(feat, label, (Xi > thres), opts)
            # Update if better solution
            if fitI < fit[i]:
                fit[i] = fitI
                X[i, :] = Xi

            # {3} Parasitism phase
            R = np.random.permutation(N)
            R = R[R != i]
            J = R[0]
            # Parasite vector
            PV = X[i, :].copy()
            # Randomly select random variables
            r_dim = np.random.permutation(dim)
            dim_no = np.random.randint(1, dim + 1)
            for d in range(dim_no):
                # Update solution
                PV[r_dim[d]] = lb + (ub - lb) * np.random.rand()
            # Boundary
            PV = np.clip(PV, lb, ub)
            # Fitness
            fitPV = fun(feat, label, (PV > thres), opts)
            # Replace parasite if it is better than j
            if fitPV < fit[J]:
                fit[J] = fitPV
                X[J, :] = PV

        # Update global best
        fitG = np.min(fit)
        idx = np.argmin(fit)
        Xgb = X[idx, :]

        curve[t] = fitG
        print('Iteration', t, 'GBest (SOS)=', curve[t])
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    SOS = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return SOS

def optim(feat, label, feat_val, label_val, opts):
    return jSymbioticOrganismsSearch(feat, label, feat_val, label_val, opts)