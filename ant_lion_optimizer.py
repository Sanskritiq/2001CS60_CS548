import numpy as np
from fitness_function import fitness_function

def ant_lion_optimizer(feat, label, feat_val, label_val, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5

    if 'T' in opts:
        max_Iter = opts['T']
    if 'N' in opts:
        N = opts['N']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    fun = fitness_function

    # Number of dimensions
    dim = feat.shape[1]

    # Initial: Ant & antlion
    Xal = np.random.uniform(lb, ub, (N, dim))
    Xa = np.random.uniform(lb, ub, (N, dim))

    # Fitness of antlion
    fitAL = np.zeros(N)
    fitE = np.inf
    for i in range(N):
        fitAL[i] = fun(feat, label, feat_val, label_val, (Xal[i, :] > thres), opts)
        # Elite update
        if fitAL[i] < fitE:
            Xe = Xal[i, :]
            fitE = fitAL[i]

    # Pre
    fitA = np.ones(N)

    curve = np.zeros(max_Iter)
    curve[0] = fitE
    t = 1

    # Iteration
    while t <= max_Iter:
        # Set weight according to iteration
        I = 1
        if t > 0.1 * max_Iter:
            w = 2
            I = (10 ** w) * (t / max_Iter)
        elif t > 0.5 * max_Iter:
            w = 3
            I = (10 ** w) * (t / max_Iter)
        elif t > 0.75 * max_Iter:
            w = 4
            I = (10 ** w) * (t / max_Iter)
        elif t > 0.9 * max_Iter:
            w = 5
            I = (10 ** w) * (t / max_Iter)
        elif t > 0.95 * max_Iter:
            w = 6
            I = (10 ** w) * (t / max_Iter)

        # Radius of ant's random walks hyper-sphere
        c = lb / I
        d = ub / I

        # Convert probability
        Ifit = 1 / (1 + fitAL)
        prob = Ifit / np.sum(Ifit)

        for i in range(N):
            # Select one antlion using roulette wheel
            rs = roulette_wheel_selection(prob)
            # Apply random walk of ant around antlion
            RA = random_walk_ALO(Xal[rs, :], c, d, max_Iter, dim)
            # Apply random walk of ant around elite
            RE = random_walk_ALO(Xe, c, d, max_Iter, dim)
            # Elitism process
            for j in range(dim):
                Xa[i, j] = (RA[t, j] + RE[t, j]) / 2
            # Boundary
            Xa[i, :] = np.clip(Xa[i, :], lb, ub)

        # Fitness
        for i in range(N):
            # Fitness of ant
            fitA[i] = fun(feat, label, feat_val, label_val, (Xa[i, :] > thres), opts)
            # Elite update
            if fitA[i] < fitE:
                Xe = Xa[i, :]
                fitE = fitA[i]

        # Update antlion position
        XX = np.concatenate((Xal, Xa), axis=0)
        FF = np.concatenate((fitAL, fitA))
        idx = np.argsort(FF)
        Xal = XX[idx[:N], :]
        fitAL = FF[:N]

        # Save
        curve[t - 1] = fitE
        print('Iteration', t, 'Best (ALO)=', curve[t - 1])
        t += 1

    # Select features based on selected index
    Sf = np.nonzero(Xe > thres)[0]
    sFeat = feat[:, Sf]

    # Store results
    ALO = {'sf': Sf, 'ff': sFeat, 'nf': len(Sf), 'c': curve, 'f': feat, 'l': label}

    return ALO


def roulette_wheel_selection(prob):
    # Cumulative summation
    C = np.cumsum(prob)
    # Random one value, most probable value [0~1]
    P = np.random.rand()
    # Route wheel
    for i in range(len(C)):
        if C[i] > P:
            return i

    return len(C)  # In case no value is selected


def random_walk_ALO(Xal, c, d, max_Iter, dim):
    # Pre
    RW = np.zeros((max_Iter + 1, dim))
    R = np.zeros(max_Iter)
    # Random walk with C on antlion
    if np.random.rand() > 0.5:
        c = Xal + c
    else:
        c = Xal - c
    # Random walk with D on antlion
    if np.random.rand() > 0.5:
        d = Xal + d
    else:
        d = Xal - d
    for j in range(dim):
        # Random distribution
        for t in range(max_Iter):
            if np.random.rand() > 0.5:
                R[t] = 1
            else:
                R[t] = 0
        # Actual random walk
        X = np.concatenate(([0], np.cumsum((2 * R) - 1)))
        # [a,b]-->[c,d]
        a = np.min(X)
        b = np.max(X)
        # Normalized
        Xnorm = (((X - a) * (d[j] - c[j])) / (b - a)) + c[j]
        # Store result
        RW[:, j] = Xnorm

    return RW

def optim(data, labels, data_val, labels_val, opts):
    return ant_lion_optimizer(data, labels, data_val, labels_val, opts)