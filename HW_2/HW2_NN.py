import numpy as np
import pandas as pd
import mlrose
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm


def ann_learning_curve_size_pre(dataset_name, X_train, y_train, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print("ann_learning_curve_size_pre", difference)

    recording_and_plotting(dataset_name, name="ann_learning_curve_size_pre",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def score_time_default(dataset_name, clf_name, clf, X_train, X_test, y_train, y_test):
    start_time = time.time()
    score = 0
    difference = 0
    for i in range(10):

        clf.fit(X_train, y_train).predict(X_test)
        score += accuracy_score(y_test, clf.fit(X_train, y_train).predict(X_test))
        end_time = time.time()
        difference += (end_time - start_time)
    txt = open('Inter_model_comparison_default.txt', 'a')
    txt.write('{}_{} score:'.format(dataset_name, clf_name))
    txt.write(str(score/10))
    txt.write("\n")
    txt.write('{}_{} time:'.format(dataset_name, clf_name))
    txt.write(str(difference/10))
    txt.write("\n\n")

if __name__=="__main__":
    '''Load and standardize data set MNIST'''

    train = np.genfromtxt('fashion-mnist_train_minor.csv', delimiter=',')[1:, :]
    test = np.genfromtxt('fashion-mnist_test_minor.csv', delimiter=',')[1:, :]

    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_test = test[:, 1:]
    y_test = test[:, 0]

    # standardize the original data - this is important but usually neglected by newbies.
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    set1_name = "mnist"

    # set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, :]
    # set2 = set2.astype(int)
    #
    # # separating set2 into X and y, then train and test
    # X2 = set2[:, :-1]
    # scaler = preprocessing.StandardScaler()
    # X2 = scaler.fit_transform(X2)
    # y2 = set2[:, -1]
    # X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    # #
    # set2_name = "ESR"



    model1 = mlrose.NeuralNetwork()
    score_time_default("MNIST", "ANN",
                       MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, alpha=0.0001, random_state=1),
                       X_train, X_test, y_train, y_test)
    # score_time_default("MNIST", "ANN",
    #                    model1,
    #                    X2_train, X2_test, y2_train, y2_test)


