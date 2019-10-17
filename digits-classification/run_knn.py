from utils import *
from sklearn.metrics import accuracy_score
from l2_distance import l2_distance
import matplotlib.pyplot as plt

LEARNING_RATE = [1,3,5,7,9]
def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels

if __name__ == '__main__':
    train_inputs, train_targets = load_train()
    test_inputs, test_targets = load_test()
    valid_inputs, valid_targets = load_valid()

    val_acc = []
    test_acc = []
    for k in LEARNING_RATE:
        v_prediction = run_knn(k, train_inputs, train_targets, valid_inputs)

        t_prediction = run_knn(k, train_inputs, train_targets,test_inputs)
        val_accuracy = accuracy_score(valid_targets, v_prediction)
        test_accuracy = accuracy_score(test_targets, t_prediction)
        val_acc.append(val_accuracy)
        test_acc.append(test_accuracy)

    fig = plt.figure()
    plt.xlabel("Learning Rate")
    plt.ylabel("Classification Rate")
    plt.plot(LEARNING_RATE, val_acc, color='tab:red',label='Validation Accuracy')
    plt.plot(LEARNING_RATE, test_acc, color='tab:blue', label='Test Accuracy')
    plt.legend()
    fig.savefig('classification_rates.png')

