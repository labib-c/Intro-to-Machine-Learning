import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    assert len(data['X']) == len(data['t'])
    p = np.arange(0, len(data['X']), 1)
    np.random.shuffle(p)
    return {'X': data['X'][p], 't': data['t'][p]}

def split_data(data, num_folds, fold):
    split_X = np.array_split(data['X'], num_folds)
    split_t = np.array_split(data['t'], num_folds)
    rest_X = np.array(split_X[0:fold-1]+split_X[fold:len(split_X)])
    rest_X_flat = rest_X.reshape(-1, rest_X.shape[-1])
    rest_t = np.array(split_t[0:fold-1]+split_t[fold:len(split_t)]).flatten()
    data_fold = {'X': np.array(split_X[fold-1]), 't': np.array(split_t[fold-1])}
    data_rest = {'X': rest_X_flat, 't': rest_t}
    return data_fold, data_rest

def train_model(data, lambd):
    # (X^T*X + lambd*I)^-1*X^T*t
    X = data['X']
    t = data['t']
    XtX = np.dot(X.transpose(), X)
    lambdaI = np.identity(XtX.shape[0])*lambd
    XTt = np.dot(X.transpose(), t)
    inverse = np.linalg.inv(XtX+lambdaI)
    return np.dot(inverse, XTt)

def predict(data, model):
    return np.dot(data['X'], model)

def loss(data, model):
    predictions = predict(data, model)
    t = data['t']
    return (np.linalg.norm(t-predictions)) / t.shape[0]

def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error

if __name__ == '__main__':
    data_train = {'X': np.genfromtxt('./data/data_train_X.csv', delimiter=','), 't': np.genfromtxt('./data/data_train_y.csv', delimiter=',')}
    data_test = {'X': np.genfromtxt('./data/data_test_X.csv', delimiter=','), 't': np.genfromtxt('./data/data_test_y.csv', delimiter=',')}

    # data = shuffle_data(data_train)
    # data_fold, data_rest = split_data(data, 10, 1)
    # model = train_model(data_fold, 5)
    # # print(data_fold['t'].shape)
    # # print(predict(data_fold, model).shape)
    # print(loss(data_fold, model))
    lambd_seq = np.arange(0.02, 1.5, 0.0296)
    train_error = []
    test_error = []
    for i in lambd_seq:
        model = train_model(data_train, i)
        train_error.append(loss(data_train, model))
        test_error.append(loss(data_test, model))
    five_fold = cross_validation(data_train, 5, lambd_seq)
    ten_fold = cross_validation(data_train, 10, lambd_seq)

    print(train_error)
    