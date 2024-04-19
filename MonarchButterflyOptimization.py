"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2015]-"Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm"
https://pdf.sciencedirectassets.com/271505/1-s2.0-S0950705115X00153/1-s2.0-S0950705115002580/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHRCLd4cpAkayLEgKZ0BeN4fP%2FIZ6EJ6jFzgkZWJqjs%2FAiEApmh1YWMBc%2FRaKQnPX6KxlLAJg9MgaKj%2FoIDXEWmkq8IqswUIExAFGgwwNTkwMDM1NDY4NjUiDB7ldXHVROG2E0sLYyqQBbAV9HzHl6csLQ%2F4YXJqJTm5rXdVgQVGfqpTrotTeT73xroXhoswraJpcmh1FCNCb%2Fk5JlQWiMWwtNaNCrH9ojxE0GFbkd%2FpEbYm3r4yMQxJjb3mcDNldWUaKT%2FAdZRl%2Bea1cHii01LykdZfb02vLy5riCj32U8tZchIMJDZRPtrU%2B3vIg2RGfx%2Ftpy%2BkVEoOIIclUjjMY1IuU4JyrVGh3cKuIJULpLozNdX7qDQ1Xe3rsoYK2%2B7MXnvqyQ%2BVPI2oZRGwetrOVa6mEFhE%2BvHrTDU8o0QifhD4%2BfzR8D1Qy6pU4ICSZ8G94OB74ZkVri1qahx4M%2FgPqr2nf1Tk8qEIZ2r9KgAWtM5SuDJkbjWdY6AbZB2joGVRy0J2UYSFf5QWCgUQHfBu%2Fk4nLzD4noUVpwraYljWYi67tr1d5sdX5hkS5pMBDQnUMRfplTm%2Bz4gS89BqTmqrYVPwZY2Mz2ZLWAFso0KfqgHMDzul06BeW%2BV2GSAm4Zz7bKrENNDeJLj0o3doKFukfuFiJgYRJUeWr%2BcAxO0hanks63Uc%2FzxwR%2FDuWwCbHEJZdOhwFDu2BOwyUNTBIEJZEcl6Y1S8p6l7wlFUeceBcL49gTYJkxwGcVSND3UMlRZVqjzTofN6a9yCIBifSXlTo1qwfcwIuzzpVuMJHESRzida11h7lbCWy2BF%2BpzITo5KS125scAiAuJUgW2QlVg9PFDSsxuiNFZkiU2%2Baj5VqPFz6Nck82MoYDtmyBrEOwf1tnya%2BYKqc7%2FlCYtT6OUk1fbrjVr7tatTtcjk5Q7V9nr3EN3T4%2BIn3Th3221Krr6vj4prpEHE1ovEbxNKd%2FCeKYjtsJ6hMTEri%2BStUaOL6oQYfvb4vjnuI8nMMr7iLEGOrEBalTh9QohBFX45RrJRal0%2BnT4rWLtYEawbQ%2BqFwawq6S%2BdhJcShm3kUTI1fxTb3IQZ4QoS%2F2SjItaxMSYIXXJSPeOFcYD0HvqIU66lhnK1Z0XQqI4Zid2mcG2DZa5R0uBL2vzkPx4BVS0ZAuRnh1lV62FTal3Tvfnu6Z7FOEW5xSaadIVdw%2BsMV6HmKpKprQ23dgLBv1%2B5XEYz4v0v4bH0GNaTXHBzBuClkLwP4d2VC0R&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T103357Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6PMWXL6N%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e6e339740674ff67faa1ca08abf22f02e4c31cddbbea02eecc6351531abd8ea4&hash=b38212701a02eac30422c52c40122d4994a89b9e7648847b6069d86342492242&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0950705115002580&tid=spdf-6d9f016a-a673-4097-a211-d489b266516c&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565901510b&rr=876c3d451f6333cc&cc=in

run with : `python3 runner.py --algo "mbo"`
"""

import numpy as np
from fitness_function import fitness_function


def jMonarchButterflyOptimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    peri = 1.2  # migration period
    p = 5 / 12  # ratio
    Smax = 1  # maximum step
    BAR = 5 / 12  # butterfly adjusting rate
    num_land1 = 4  # number of butterflies in land 1
    beta = 1.5  # levy component

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'peri' in opts:
        peri = opts['peri']
    if 'p' in opts:
        p = opts['p']
    if 'Smax' in opts:
        Smax = opts['Smax']
    if 'BAR' in opts:
        BAR = opts['BAR']
    if 'beta' in opts:
        beta = opts['beta']
    if 'N1' in opts:
        num_land1 = opts['N1']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.rand(N, dim) * (ub - lb) + lb

    fit = np.zeros(N)
    fitG = np.inf

    for i in range(N):
        # Fitness
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Global best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    Xnew = np.zeros((N, dim))
    Fnew = np.zeros(N)

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 2

    while t <= max_Iter:
        # Sort butterfly
        idx = np.argsort(fit)
        X = X[idx, :]
        fit = fit[idx]

        # Weight factor
        alpha = Smax / (t ** 2)

        # First land: Migration operation
        for i in range(num_land1):
            for d in range(dim):
                # Random number
                r = np.random.rand() * peri
                if r <= p:
                    # Random select a butterfly in land 1
                    r1 = np.random.randint(0, num_land1)
                    # Update position
                    Xnew[i, d] = X[r1, d]
                else:
                    # Random select a butterfly in land 2
                    r2 = np.random.randint(num_land1, N)
                    # Update position
                    Xnew[i, d] = X[r2, d]
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

        # Second land: Butterfly adjusting operation
        for i in range(num_land1, N):
            # Levy distribution
            dx = jLevyDistribution(beta, dim)
            for d in range(dim):
                if np.random.rand() <= p:
                    # Position update
                    Xnew[i, d] = Xgb[d]
                else:
                    # Random select a butterfly in land 2
                    r3 = np.random.randint(num_land1, N)
                    # Update position
                    Xnew[i, d] = X[r3, d]
                    # Butterfly adjusting
                    if np.random.rand() > BAR:
                        Xnew[i, d] += alpha * (dx[d] - 0.5)
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

        # Combine population
        for i in range(N):
            # Fitness
            Fnew[i] = fun(feat, label, (Xnew[i, :] > thres), opts)
            # Global best update
            if Fnew[i] < fitG:
                fitG = Fnew[i]
                Xgb = Xnew[i, :]

        # Merge & Select best N solutions
        XX = np.vstack((X, Xnew))
        FF = np.hstack((fit, Fnew))
        idx = np.argsort(FF)
        X = XX[idx[:N], :]
        fit = FF[:N]

        # Save
        curve[t - 1] = fitG
        print('\nIteration %d Best (MBO)= %f' % (t, curve[t - 1]))
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    MBO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return MBO


# Levy Flight
def jLevyDistribution(beta, dim):
    # Sigma
    nume = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    # Parameter u & v
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    # Step
    step = u / np.abs(v) ** (1 / beta)
    return step


def optim(feat, label, feat_val, label_val, opts):
    return jMonarchButterflyOptimization(feat, label, feat_val, label_val, opts)