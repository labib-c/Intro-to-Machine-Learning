import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt
import statistics

LAMBDAS = [0, 0.001, 0.01, 0.1, 1.0]
def run_logistic_regression(reg):
    train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.01,
                    'weight_regularization': reg,
                    'num_iterations': 100
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.rand(M+1, 1)*0.3

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    train = []
    val = []
    train_err = []
    val_error = []
    test_err = []
    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        predictions_test = logistic_predict(weights, test_inputs)
        ce_test, frac_correct_test = evaluate(test_targets, predictions_test)
        test_err.append(1-frac_correct_test)
        
        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        train.append(cross_entropy_train)
        val.append(cross_entropy_valid)
        train_err.append(1-frac_correct_train)
        val_error.append(1-frac_correct_valid)
    # fig = plt.figure()
    # plt.xlabel("Iteration")
    # plt.ylabel("Cross Entropy")
    # plt.plot(train, color='tab:blue', label="Train")
    # plt.plot(val, color='tab:red',label="Validation")
    # plt.legend()
    # fig.savefig("2_3__reg"+str(reg).replace(".","_")+".png")
    return statistics.mean(train), statistics.mean(train_err), statistics.mean(val), statistics.mean(val_error), test_err


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic_pen,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)

if __name__ == '__main__':
    total_val_ce = [[],[],[],[],[]]
    total_val_err = [[],[],[],[],[]]
    total_train_ce = [[],[],[],[],[]]
    total_train_err = [[],[],[],[],[]]
    for l in range(len(LAMBDAS)):
        for i in range(0, 5):
            train_ce, train_err, val_ce, val_err, test = run_logistic_regression(LAMBDAS[l])
            total_val_ce[l].append(val_ce)
            total_val_err[l].append(val_err)
            total_train_ce[l].append(train_ce)
            total_train_err[l].append(train_err)
    avg_v_ce = []
    avg_t_ce = []
    avg_v_err = []
    avg_t_err = []
    for j in range(len(LAMBDAS)):
        avg_v_ce.append(sum(total_val_ce[j])/len(total_val_ce[j]))
        avg_t_ce.append(sum(total_train_ce[j]) / len(total_train_ce[j]))
        avg_v_err.append(sum(total_val_err[j]) / len(total_val_err[j]))
        avg_t_err.append(sum(total_train_err[j]) / len(total_train_err[j]))

    best_lambda = LAMBDAS[avg_v_err.index(min(avg_v_err))]
    print(best_lambda)
    fig = plt.figure()
    plt.xlabel("Weight Regularizers")
    plt.ylabel("CE")
    plt.plot(LAMBDAS, avg_v_ce, label="Validation")
    plt.plot(LAMBDAS, avg_t_ce, label='Train')
    plt.legend()
    fig.savefig("2_3_small_avg_ce_act")

    # train_ce, train_err, val_ce, val_err, test_err = run_logistic_regression(best_lambda)
    # print("TEST ERROR FOR {}: {}".format(best_lambda, statistics.mean(test_err)))