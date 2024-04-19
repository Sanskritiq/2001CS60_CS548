"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2019]-"Poor and rich optimization algorithm: A new human-based and multi populations algorithm"
https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197619X00081/1-s2.0-S0952197619302167/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCN%2BEQWpi7Tw1B7YHcZIU3YEoeRV5ZKv41EX%2FGT82cXZAIhAMMjVmwefTZsKAnQqY3N26JmpTS%2F7cdA4cNhjZLS6bCOKrMFCBMQBRoMMDU5MDAzNTQ2ODY1IgxLy%2F%2BKanp%2BMY3jJ2wqkAWOXIWtJ8E3PW9or85Ic7esduj0LtufGmC8UMjRg0AzLi0Jsj7Q1jiJ%2Fkb%2BIk12uukoFFs1PzSCSiLl%2FnrC1E%2Br8rxzqK%2Bz8CK%2BCaemTZWjqNgInsZQbSKDCNUM0Mp0tpROLRHFy6kpQLs9eUY1HPYlQOEkm9FbQwNI%2FgPYSIDy52hpjH4BYi%2FVi8JioRz8GhvaflF%2FGD%2F2wLrRHAnxbenTCCJYF0QHvYKvU55OqnytmzTIqt1WHOiVq%2B1mwQ3adR9ZqAFWTQWvDSYF4hIRbT%2FYde889idUaHKwRilkUf0vqb7qRgHZz%2B9eq8yBN5SbRgq0IL1ULFtaVtGUpgF%2ByuI95rT1i5xYE916In89%2FgeMDcNCk9zQVkwkVOAFg%2FmjNcESzyUDpxjPJxiuYfC5%2BGfJIF1fnNTklA567xDHIOi9gwMnHdoLurncXlNDf2Pqr01Uricx1cb0CNIL%2FeytCp8Rs6w5W%2B9qwqM0lx2WJ3iCA6c3XEEYfZP1nPNeHlFi7CCo446Hy2SN8nAc9cX%2B2mAaM4l%2FtDEKuDZuoXL7G6hwkMY2JfcYiaHxgAKEq8Pren7tmLWStnpWpmm90uqY3Rs9MtSlI8xumaa%2FALiAy%2FORmzjOKYEVKi3t%2Bg9%2Btv5cGkcTf0aMLWnwVL%2Bz%2F%2Fa5ZCA6z%2FYPNP60u4VZ6dhdK3iLLxF51uYBhTPHki9MuZA0ah1%2B05L%2Bsnf%2Fo%2FwtpTEQd1hS5y0spYAI12Cl%2BEDALM5oBZ3ZQIX2lw6pnmpWcoGtFTEmugBF%2Fl2CIMK4G3Yfovu78vEY0hUCz3lVz%2BlVjIigIcZeIKekspgSexU4i0tt1KYY2t7ufqg46PZvhwjkePkve2its%2BFgeaF1mTz7HOm4XTCR%2B4ixBjqwAWt0VYfuCL0%2BkIStYZfATQ0Rs2lLpIQE9L6oj6X8c8PgkCMhP5K4WZn4IVDURC63NweY5YupxcDubaIVfqxJmwP8cSn3pH5QNEhAN%2BwAMZPwTRRTSfg24uR9yZGpoR44Ea%2FaADjRkSOMZfvlYf6K37ibRMCdih25Pb70pNawWVxC2%2BYmo8UXBV%2BJ%2BbVWWnDZc7bb58gVA6MpSdGW42%2FrBX8MQWgUkEnGR03TJQOWYjiR&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T101503Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYR73LSZ3P%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=5f21b3d103748d86a02d3e4fb9f6ec213876e826e17600f6b1e305d2186164f0&hash=4617037c777fe6c29d0c603f212df291ddd8ce9f10c9f160ac48c43f937da9db&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197619302167&tid=spdf-9ff15d87-725e-49fb-8c8c-094083ac0d42&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565901505e&rr=876c21953d7f33b4&cc=in

run with : `python3 runner.py --algo "pro"`
"""

import numpy as np
from fitness_function import fitness_function


def jPoorAndRichOptimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    Pmut = 0.06  # mutation probability

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_Iter = opts["T"]
    if "Pmut" in opts:
        Pmut = opts["Pmut"]
    if "thres" in opts:
        thres = opts["thres"]

    # Double population size: Main = Poor + Rich
    N = N + N

    # Objective function
    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.rand(N, dim) * (ub - lb) + lb

    # Fitness
    fit = np.zeros(N)
    fitG = np.inf

    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Sort poor & rich
    idx = np.argsort(fit)
    fit = fit[idx]
    X = X[idx, :]

    # Pre
    XRnew = np.zeros((N // 2, dim))
    XPnew = np.zeros((N // 2, dim))
    fitRnew = np.zeros(N // 2)
    fitPnew = np.zeros(N // 2)

    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1

    # Iteration
    while t < max_Iter:
        # Divide poor & rich
        XR = X[: N // 2, :]
        fitR = fit[: N // 2]
        XP = X[N // 2 :, :]
        fitP = fit[N // 2 :]

        # Select best rich individual
        idxR = np.argmin(fitR)
        XR_best = XR[idxR, :]

        # Select best poor individual
        idxP = np.argmin(fitP)
        XP_best = XP[idxP, :]

        # Compute mean of rich
        XR_mean = np.mean(XR, axis=0)

        # Compute worst of rich
        idxW = np.argmax(fitR)
        XR_worst = XR[idxW, :]

        # [Rich population]
        for i in range(N // 2):
            for d in range(dim):
                # Generate new rich
                XRnew[i, d] = XR[i, d] + np.random.rand() * (XR[i, d] - XP_best[d])

                # Mutation
                if np.random.rand() < Pmut:
                    # Mutation
                    G = np.random.randn()
                    XRnew[i, d] += G

            # Boundary
            XRnew[i, :] = np.clip(XRnew[i, :], lb, ub)

            # Fitness of new rich
            fitRnew[i] = fun(feat, label, (XRnew[i, :] > thres), opts)

        # [Poor population]
        for i in range(N // 2):
            for d in range(dim):
                # Calculate pattern
                pattern = (XR_best[d] + XR_mean[d] + XR_worst[d]) / 3

                # Generate new poor
                XPnew[i, d] = XP[i, d] + (np.random.rand() * pattern - XP[i, d])

                # Mutation
                if np.random.rand() < Pmut:
                    # Mutation
                    G = np.random.randn()
                    XPnew[i, d] += G

            # Boundary
            XPnew[i, :] = np.clip(XPnew[i, :], lb, ub)

            # Fitness of new poor
            fitPnew[i] = fun(feat, label, (XPnew[i, :] > thres), opts)

        # Merge all four groups
        X = np.vstack((XR, XP, XRnew, XPnew))
        fit = np.hstack((fitR, fitP, fitRnew, fitPnew))

        # Select the best N individual
        idx = np.argsort(fit)
        fit = fit[idx][:N]
        X = X[idx[:N], :]

        # Best update
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0, :]

        curve[t] = fitG
        print("Iteration %d Best (PRO)= %f" % (t, curve[t]))
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    PRO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return PRO


def optim(feat, label, feat_val, label_val, opts):
    return jPoorAndRichOptimization(feat, label, feat_val, label_val, opts)