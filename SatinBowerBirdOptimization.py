"""
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2017]-"Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation"

https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197617X0002X/1-s2.0-S0952197617300064/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIACKIKz3BH7dVll2tZPwT15E1oQY8pFWM1DgjTHBUQtGAiEAkDtgWHWZZUVdkGJZAhT8lihzdXNTXoR5h571ly81%2FAQqswUIEhAFGgwwNTkwMDM1NDY4NjUiDKinLfexQbZ4GAt0vSqQBc%2B1PgLCnFa67Kx9pDKVFxJqydIH7gM2DyC6OyOz8TuyvXnxFO0hMdvKBgqAQDOMhV8pBIpNElzwvFnVDEXuFkrEU76HClxEA%2BUWxJxxKGjO9tF9HP%2BeMBggLd%2FmLyJKI%2F5zeUcqeAte4CVrtH5uJOLHn7qY1J%2Fw3DI3jARbHgzgtH4d%2FkiZx5qrfb%2FsQynk%2FJsifnDkpEfiWaZRVJjp1rLPt44UOGwsQYFdelV9UcurjNbN5LW6USBo4eeeMwszUu58cbCNvt4TY9k2BGe4sw0EfW59VjTjpvM5FeUVV%2B4gVf%2F16zErjChsIrejhDlCJDbphn46203oe4IsiUP85OnGxWyilAmVCYTHTKfOenMSPQQJhQHggMDX7aw929xXkWUJlBWlRtOHG3o27GpcZVq18vudMgEQ2c09eaR%2B1B1TaFQioDffdaI3vWIOGCTHgZHe5%2BkLqTiMpCLstSds1HEapDCkeF0rYcDRFnqvnzrDJNIr5rm4wJLnDLqx0NjKWnwugBI5H2C%2B6rxUTf4oUIDxPvzbe%2Fxyb8EAFodGnfue%2BP9Qt%2FIVPVKiinxS33Ow4C9AMkyoiZ9XpbcH9LD67rq7OHj6tpKN2WWpQFamat0DjA8l2WLb3AlTo0cIERVa4LnNv1h1dm%2FAXtIVXebZTAiODY02kIx13%2FVm7w5%2FkdINtJ%2B0etloYm%2BwOGao5fDS38M71DLqWplMZwr2v8Ej5BZhHna%2BOWC2XnzI5LNQOma5YFNuxv80HixyAipfwPiNm3Pyeot5WiXT19qEdrC%2Boqv1wl7iPhGSIgYVVGq6H6y1znWn5qoIbJwCjtULryiCgK3fokQQblaeObNGY2LC%2BVt8zg4OptZU%2BNQPsQG9RXxFMLXsiLEGOrEB8eVP0tUamZyF1BtlX5oiEBpkOeXL8XjzijAzMOHKs6ZlLUt5SNDSZpJwaKpg7Yg2sHWy3TXUzVAVGKC8ygXVectS%2BQjZQ8OnfOt6UPECuwOZKqQHLi%2FrmkRSijkGyOEvP7ujNE8s9WzNENrP%2F0UPRESOCr4vYlFwF7%2FyRrGSA6wR2CY7QsR7TcG4FiAmdgKeCyMe7zNqp4fw0Mn3GBR6xmpwLIIb8JIURqE6jeLUXtx5&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T101207Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZGTTERKQ%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4eaace684fa2dcf2adf1d54ae6cba668aa8fd3347b25a5b4b435e332ba419dfa&hash=9257beaf884f57f7f5ac2fbf6c39d5be50fb4615e563d35aa0f5a12b1f4d4696&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197617300064&tid=spdf-290c3410-1e00-4cca-8f3f-7254f9121fb6&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565901530b&rr=876c1d493bed33b4&cc=in

run with : `python3 runner.py --algo "sbo"`
"""

import numpy as np
from fitness_function import fitness_function

def jSatinBowerBirdOptimization(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    alpha = 0.94   # constant
    z = 0.02   # constant
    MR = 0.05   # mutation rate

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'z' in opts:
        z = opts['z']
    if 'MR' in opts:
        MR = opts['MR']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    def fun(feats, labels, X, options):
        # Define your fitness function here
        return fitness_function(feats, labels, feat_val, label_val, X, options)

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, size=(N, dim))

    # Fitness
    fit = np.zeros(N)
    fitE = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Elite update
        if fit[i] < fitE:
            fitE = fit[i]
            Xe = X[i, :]

    # Sigma (7)
    sigma = z * (ub - lb)

    # Pre
    Xnew = np.zeros((N, dim))
    Fnew = np.zeros(N)

    curve = np.zeros(max_Iter + 1)
    curve[0] = fitE
    t = 1

    # Iterations
    while t <= max_Iter:
        # Calculate probability (1-2)
        Ifit = 1 / (1 + fit)
        prob = Ifit / np.sum(Ifit)
        for i in range(N):
            for d in range(dim):
                # Select a bower using roulette wheel
                rw = jRouletteWheelSelection(prob)
                # Compute lambda (4)
                lambd = alpha / (1 + prob[rw])
                # Update position (3)
                Xnew[i, d] = X[i, d] + lambd * (((X[rw, d] + Xe[d]) / 2) - X[i, d])
                # Mutation
                if np.random.rand() <= MR:
                    # Normal distribution & Position update (5-6)
                    r_normal = np.random.randn()
                    Xnew[i, d] = X[i, d] + (sigma * r_normal)
            # Boundary
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)
        # Fitness
        for i in range(N):
            Fnew[i] = fun(feat, label, (Xnew[i, :] > thres), opts)
        # Merge & Select best N solutions
        XX = np.vstack((X, Xnew))
        FF = np.hstack((fit, Fnew))
        idx = np.argsort(FF)
        X = XX[idx[:N], :]
        fit = FF[:N]
        # Elite update
        if fit[0] < fitE:
            fitE = fit[0]
            Xe = X[0, :]
        # Save
        curve[t] = fitE
        print('Iteration', t, 'Best (SBO)=', curve[t])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xe > thres]
    sFeat = feat[:, Sf]

    # Store results
    SBO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return SBO

# Roulette Wheel Selection 
def jRouletteWheelSelection(prob):
    # Cummulative summation
    C = np.cumsum(prob)
    # Random one value, most probability value [0~1]
    P = np.random.rand()
    # Route wheel
    for i in range(len(C)):
        if C[i] > P:
            Index = i
            break
    return Index


def optim(feat, label, feat_val, label_val, opts):
    return jSatinBowerBirdOptimization(feat, label, feat_val, label_val, opts)