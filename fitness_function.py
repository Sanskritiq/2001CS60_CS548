from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def fitness_function(feat, label, feat_val, label_val, X, opts):
    # Default values for alpha and beta
    ws = [0.99, 0.01]

    if 'ws' in opts:
        ws = opts['ws']

    # Check if any feature exists
    if np.sum(X == 1) == 0:
        cost = 1
    else:
        # Error rate
        error = wrapper_KNN(feat[:, X == 1], label, feat_val[:, X == 1], label_val, opts)
        # Number of selected features
        num_feat = np.sum(X == 1)
        # Total number of features
        max_feat = len(X)
        # Set alpha & beta
        alpha = ws[0]
        beta = ws[1]
        # Cost function
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost

def wrapper_KNN(sFeat, label, sFeat_val, label_val, opts):
    k = 5  # Default value for k
    if 'k' in opts:
        k = opts['k']
    # if 'Model' in opts:
    #     Model = opts['Model']

    # # Define training & validation sets
    # trainIdx = Model.train
    # testIdx = Model.test
    xtrain = sFeat
    ytrain = label
    xvalid = sFeat_val
    yvalid = label_val

    # Training model
    My_Model = KNeighborsClassifier(n_neighbors=k)
    My_Model.fit(xtrain, ytrain)

    # Prediction
    pred = My_Model.predict(xvalid)

    # Accuracy
    Acc = np.sum(pred == yvalid) / len(yvalid)

    # Error rate
    error = 1 - Acc

    return error
