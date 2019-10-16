""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid, load_valid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    # data = valid_inputs
    # z = w^Tx +b
    # y = sigmoid(z)

    x = np.c_[data, np.ones(data.shape[0])]
    z = np.dot(x, weights)
    y = sigmoid(z)

    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    ce = -np.sum(targets*np.logaddexp(0, y) + (1-targets)*np.logaddexp(0, 1-y))
    apply_thresh = (y > 0.5)
    frac_correct = (targets == apply_thresh).mean()
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    x = np.c_[data, np.ones(data.shape[0])]
    f, frac_correct = evaluate(targets, y)
    df = x.transpose().dot(y - targets)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    f, df, y = logistic(weights, data, targets, hyperparameters)
    # TODO: exlcude last row of weights somehow
    f = f + (hyperparameters['weight_regularization']/2)*np.sum(weights**2)
    df = df + hyperparameters['weight_regularization']*weights
    return f, df, y

if __name__ == '__main__':
    valid_inputs, valid_targets = load_valid()
    weights = np.random.rand(valid_inputs.shape[1]+1, 1)
    y = logistic_predict(weights, valid_inputs)
    ce, frac_correct = evaluate(valid_targets, y)
    f, df, y = logistic(weights, valid_inputs, valid_targets, None)

