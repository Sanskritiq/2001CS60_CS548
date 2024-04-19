

'''2001CS57
Output:
    Acc : accuracy of validation model
    FS : feature selection model ( It contains several results )
        sf : index of selected features
        ff : selected features
        nf : number of selected features
        c : convergence curve
        t : computational time (s)

[2015]-"TSA: Tree-seed algorithm for continuous optimization

https://pdf.sciencedirectassets.com/271506/1-s2.0-S0957417415X00121/1-s2.0-S0957417415002973/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIAD2Xrou%2BmxZ81REEkpRS%2F2bTubnXXfIWCr6U9Ro7G5SAiEAyHv4D4sfbn%2BnQII0APufQiTm7nAlmqW9PbnM%2Fj9Ae7IquwUI%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDAOSTEnQbZ5Qvn03GyqPBR%2BI7QOfNXXsTmHKSTfwRIPoxIT03UDmrv1ShSa3hy5dbvT61BY3%2FFuLprnlxYud68ij3RGcP%2FJhROa%2BiaUN08jMm6cy87drgEd0bAW716J8lGfp30uiU8zMZPC0HxJVEVUYVcIO1PR6k3IjXJOZOyg%2BrEMNzz6iwbEtUpJ1dN0zY1gH2BVTP6vfshShw4rl1RFy%2BywC1oflkd0zlbFSmIYkuI90EMbqCkrofVwPdwcvCSNKgSIrC%2FxCCT0C0Fl5EmU3hXZYhuDpw8knTZNsQMxpLKOmxsZe%2BjgxxVwvvQ4IQdZyEi8qMnEAiqVm8JXWI1ZNEnS%2BI5sXxq46RsiocwapWhZQ12ri%2FROxsVa8OQ4Ie9nFg%2BZ%2B3vIEvtRsdLG9MJYD2CUn7Bb67sEEHQ06lvSj7kymfJMZgJCysmpmQxEuc56MadrrvJxrSrW8OYKQbhyFnMVFhlJqhEsmMfqZJktFDq5WuAZzsip2gVoNYn5eiJ%2FG3SLssV%2FoIsGGHZ4CwgrTRPVni4oerMUpj8p4f9fH4%2F2P8zYqZmWDnPBCrAHNYQGrIOjZ4ojOJ6BIpfqlwn28xeM5nWFIH9cq0PSic1gI9f61SKQ%2BjFZJgU2lg7woPCxLP3VoqfcjUsIANxkNCT3rLVngAFwIlqh6DbqVg7YCYxzzVAMnaD62berLPx813yyxSQ2qGE1dM8JtHb2HhAaDAK%2Bak1UnX%2FJwH0FEokmeBnwLSeMofKXCCP4DCoQKPu8M4R1epPNcW0unQnwteX1HHUp%2Bnhqs741bq6VLG6aO%2BjrRipfEmnNVZinD2DVTKKv6f6LdYA01Qahztr7K%2FFlqOnDTJ46McijxLHu4afklo8%2BIXy%2BOtqzzcuUK70wwppaIsQY6sQF1GFB3W4wRXU9NGxkZ4oTY8Py5roDCQ96Jw1m%2FLkV%2B4tq2LdbH%2BFZtTOJRGAdX7Q2ghjnNnYaK3hcysoTv74EsMatjwEFceR7yByqw0U8xyhy4kJCJPeAvnDJzzD9lGCPrjpVfnH0%2F2GAd1V2K1dfk%2FlNKghm5wQfG8eLjTcJxJY0v%2FTKXl4KnAswtHTaQl8I%2FL3e0Bdnz2GVk4ZrHeRFprI1PkFHXb79JFlFrGK%2FpKW4%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240419T072300Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTYXSF4JEXP%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8004b68872cce924cfdf501308a7237a25c1f5a899ff8817a401095fdc4d6e0f&hash=c2706fd01ee6d86b4ce5cff4ea4dd11fe8e9f978633bd40184e6374c93b4a5a4&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0957417415002973&tid=spdf-e5f9026e-3b1f-400e-997e-6d003f5545c2&sid=95d572e41c901749886a19337355a8a081e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=090b5e565900505a&rr=876b258e2e563384&cc=in

'run with : `python3 runner.py --algo "tsa"`
'''

import numpy as np
from fitness_function import fitness_function

def jTreeSeedAlgorithm(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    ST = 0.1  # switch probability

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'ST' in opts:
        ST = opts['ST']
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

    # Maximum & minimum number of seed
    Smax = round(0.25 * N)
    Smin = round(0.1 * N)

    # Pre
    curve = np.zeros(max_Iter)
    curve[0] = fitG
    t = 1

    # Iteration
    while t < max_Iter:
        for i in range(N):
            # Random number of seed
            num_seed = round(Smin + np.random.rand() * (Smax - Smin))
            Xnew = np.zeros((num_seed, dim))
            for j in range(num_seed):
                # Random select a tree, but not i
                RN = np.random.permutation(N)
                RN = RN[RN != i]
                r = RN[0]
                for d in range(dim):
                    # Alpha in [-1,1]
                    alpha = -1 + 2 * np.random.rand()
                    if np.random.rand() < ST:
                        # Generate seed
                        Xnew[j, d] = X[i, d] + alpha * (Xgb[d] - X[r, d])
                    else:
                        # Generate seed
                        Xnew[j, d] = X[i, d] + alpha * (X[i, d] - X[r, d])
                # Boundary
                Xnew[j, :] = np.clip(Xnew[j, :], lb, ub)

            # Fitness
            for j in range(num_seed):
                # Fitness
                Fnew = fun(feat, label, (Xnew[j, :] > thres), opts)
                # Greedy selection
                if Fnew < fit[i]:
                    fit[i] = Fnew
                    X[i, :] = Xnew[j, :]

        # Best solution
        fitG_new = np.min(fit)
        idx = np.argmin(fit)
        Xgb_new = X[idx, :]

        # Best update
        if fitG_new < fitG:
            fitG = fitG_new
            Xgb = Xgb_new

        # Store
        curve[t] = fitG
        print('Iteration', t, 'Best (TSA)=', curve[t])
        t += 1

    # Select features
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # Store results
    TSA = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return TSA


def optim(feat, label, feat_val, label_val, opts):
    return jTreeSeedAlgorithm(feat, label, feat_val, label_val, opts)