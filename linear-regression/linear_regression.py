import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    assert data['X'].shape[0] == data['t'].shape[0]
    p = np.arange(0, len(data['X']), 1)
    np.random.shuffle(p)
    return {'X': data['X'][p], 't': data['t'][p]}

def split_data(data, num_folds, fold):
    split_X = np.array_split(data['X'], num_folds)
    split_t = np.array_split(data['t'], num_folds)

    fold_X = split_X.pop(fold-1)
    fold_t = split_t.pop(fold-1)

    data_fold = {'X': np.array(fold_X), 't': np.array(fold_t)}
    data_rest = {'X': np.vstack(split_X), 't': np.array(split_t).flatten()}
    return data_fold, data_rest

def train_model(data, lambd):
    # (X^T*X + lambd*I)^-1*X^T*t

    X = data['X']
    t = data['t']
    XtX = np.dot(X.transpose(), X)
    lambdaI = np.identity(XtX.shape[0])*lambd
    XtX_sum_lambda = XtX + lambdaI
    first_term = np.linalg.inv(XtX_sum_lambda)
    XTt = X.transpose().dot(t)
    return first_term.dot(XTt)

def predict(data, model):
    return np.dot(data['X'], model)

def loss(data, model):
    predictions = predict(data, model)
    t = data['t']
    return (np.linalg.norm(t-predictions))**2 / t.shape[0]

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

    lambd_seq = np.arange(0.02, 1.5, 0.03)
    train_error = []
    test_error = []
    for i in lambd_seq:
        model = train_model(data_train, i)
        train_error.append(loss(data_train, model))
        test_error.append(loss(data_test, model))
    five_fold = cross_validation(data_train, 5, lambd_seq)
    ten_fold = cross_validation(data_train, 10, lambd_seq)

    min_idx_five = five_fold.index(min(five_fold))
    min_idx_ten = ten_fold.index(min(ten_fold))
    train_idx = train_error.index(min(train_error))
    test_idx = test_error.index(min(test_error))

    print("The minimum λ for five-fold CV is {}".format(lambd_seq[min_idx_five].round(4)) )
    print("The minimum λ for ten-fold CV is {}".format(lambd_seq[min_idx_ten].round(4)))
    print("The minimum λ for training error is {}".format(lambd_seq[train_idx].round(4)))
    print("The minimum λ for test error is {}".format(lambd_seq[test_idx].round(4)))
    # fig = plt.figure(figsize=(10,5))
    # plt.xticks(np.arange(0.02, 1.5, 0.1))
    # plt.xlabel("Lambda Sequence")
    # plt.ylabel("Errors")
    # plt.plot(lambd_seq, train_error, color='tab:blue', label='training error')
    # plt.plot(lambd_seq, test_error, color='tab:green', label='test error')
    # plt.plot(lambd_seq, five_fold, color='tab:red', label='5-fold CV')
    # plt.plot(lambd_seq, ten_fold, color='tab:orange', label='10-fold CV')
    # plt.legend()
    # fig.savefig('cv.png')