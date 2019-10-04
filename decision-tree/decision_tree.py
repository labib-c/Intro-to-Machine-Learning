from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from os import system
import pydotplus
import numpy as np
import pandas as pd
import random
import math

FAKE_DATA = './data/clean_fake.txt'
REAL_DATA = './data/clean_real.txt'
DEPTHS = [150,311,625,1250,2500]

vectorizer = CountVectorizer()

def label_and_shuffle_data():
    data = []
    fake = open(FAKE_DATA, 'r')
    for f in fake:
        d = (f.strip(), 'fake') #fake
        data.append(d)
    real = open(REAL_DATA, 'r')
    for r in real:
        d = (r.strip(), 'real') #real
        data.append(d)
    random.shuffle(data)

    return pd.DataFrame(data, columns=['headline', 'label'])

def load_data():
    data = label_and_shuffle_data()
    X = vectorizer.fit_transform(data['headline'])

    X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.3, random_state=1)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def select_model():
    # for information gain, criterion='entropy'
    # for Gini, criterion='gini'
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()
    best_model = (None, None)
    best_acc = 0
    for d in DEPTHS:
        clf = DecisionTreeClassifier(max_depth=d, criterion='gini')
        clf.fit(X_train, y_train)
        acc = validation_accuracy(clf.predict(X_val), y_val.tolist())
        if acc > best_acc:
            best_acc = acc
            best_model = ('gini', d)
        print("Accuracy ({}, {}): {}".format('gini', d, acc))
    
    for d in DEPTHS:
        clf = DecisionTreeClassifier(max_depth=d, criterion='entropy')
        clf.fit(X_train, y_train)
        acc = validation_accuracy(clf.predict(X_val), y_val.tolist())
        if acc > best_acc:
            best_acc = acc
            best_model = ('entropy', d)
        print("Accuracy ({}, {}): {}".format('entropy', d, acc))
    
    clf = DecisionTreeClassifier(max_depth=best_model[1], criterion=best_model[0])
    clf.fit(X_train, y_train)
    make_tree_image(clf)
    return best_model
    

def validation_accuracy(prediction, target):
    assert len(prediction) == len(target)
    total_mistake = 0
    for i in range(len(prediction)):
        total_mistake += 1 if prediction[i] == target[i] else 0
    return total_mistake/len(prediction)

def make_tree_image(clf):
    dotfile = open("./dtree.dot", 'w')
    export_graphviz(clf, out_file = dotfile, filled=True, rounded=True, class_names=clf.classes_,
                        special_characters=True, max_depth=2, feature_names=vectorizer.get_feature_names())
    
    dotfile.close()      

    # make sure you have Graphviz installed
    # sudo apt-get install graphviz
              
    from subprocess import call
    call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])
    
def compute_informatian_gain(Y, x_i):
    return get_entropy(Y) - get_conditional_entropy(Y, x_i) 

def get_entropy(Y):
    real = Y[Y['label'] == 'real'].count()['label']
    fake = Y[Y['label'] == 'fake'].count()['label']
    total = real+fake
    print(real)
    print(fake)
    p_real = real / total
    p_fake = fake/ total
    return -(p_real*log(p_real, 2) + p_fake*log(p_fake, 2))

def log(prob, base):
    if prob == 0.0:
        return 0
    else:
        return math.log(prob)
def get_conditional_entropy(Y, x_i):
    pattern = '\\b'+x_i+'\\b' #regex to get exact match rather than contains
    contains_xi = Y[Y['headline'].str.contains(pattern, regex = True)]
    c_real = contains_xi[contains_xi['label'] == 'real'].count()['label']
    c_fake = contains_xi[contains_xi['label'] == 'fake'].count()['label']
    c_total = c_real + c_fake
    c_entropy = 0
    if c_total > 0:
        c_entropy = -(c_real/c_total*log(c_real/c_total, 2) + c_fake/c_total*log(c_fake/c_total, 2))

    without_xi = Y[~Y['headline'].str.contains(pattern, regex = True)]
    w_real = without_xi[without_xi['label'] == 'real'].count()['label']
    w_fake = without_xi[without_xi['label'] == 'fake'].count()['label']
    w_total = w_real + w_fake
    w_entropy = 0
    if w_total > 0:
        w_entropy = -(w_real/w_total*log(w_real/w_total, 2) + w_fake/w_total*log(w_fake/w_total, 2))
    return (c_total/(c_total+w_total))*c_entropy + (w_total/(c_total+w_total))*w_entropy

if __name__ == "__main__":
    # print("Best model: {}".format(select_model()))
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()
    Y = label_and_shuffle_data()
    print(get_entropy(y_train))
    x_i = 'the'
    print("IG:{}, word: {}".format(compute_informatian_gain(Y, x_i), x_i))



