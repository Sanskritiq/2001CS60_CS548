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

run with : `python3 runner.py --algo "mfo"`
"""

import numpy as np
from fitness_function import fitness_function


def jMothFlameOptimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    b = 1  # constant

    if "T" in opts:
        max_Iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "b" in opts:
        b = opts["b"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.rand(N, dim) * (ub - lb) + lb

    curve = np.inf * np.ones(max_Iter+1)
    t = 1

    while t <= max_Iter:
        fit = np.zeros(N)
        fitG = np.inf

        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Global best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        if t == 1:
            # Best flame
            idx = np.argsort(fit)
            fitF = fit[idx]
            flame = X[idx, :]
        else:
            # Sort population
            XX = np.vstack((flame, X))
            FF = np.hstack((fitF, fit))
            idx = np.argsort(FF)
            flame = XX[idx[:N], :]
            fitF = FF[:N]

        # Flame update
        flame_no = int(round(N - t * ((N - 1) / max_Iter)))
        r = -1 + t * (-1 / max_Iter)

        for i in range(N):
            # Normal position update
            if i < flame_no:
                for d in range(dim):
                    # Parameter T0
                    T = (r - 1) * np.random.rand() + 1
                    # Distance between flame & moth
                    dist = np.abs(flame[i, d] - X[i, d])
                    # Moth update
                    X[i, d] = dist * np.exp(b * T) * np.cos(2 * np.pi * T) + flame[i, d]
            # Position update respect to best flames
            else:
                for d in range(dim):
                    # Parameter T
                    T = (r - 1) * np.random.rand() + 1
                    # Distance between flame & moth
                    dist = np.abs(flame[i, d] - X[i, d])
                    # Moth update
                    X[i, d] = (
                        dist * np.exp(b * T) * np.cos(2 * np.pi * T)
                        + flame[flame_no - 1, d]
                    )

            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        curve[t - 1] = fitG
        print("Iteration %d Best (MFO)= %f" % (t, curve[t - 1]))
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    MFO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return MFO


def optim(feat, label, feat_val, label_val, opts):
    return jMothFlameOptimization(feat, label, feat_val, label_val, opts)