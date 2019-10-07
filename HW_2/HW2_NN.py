import numpy as np
import pandas as pd
import mlrose
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm

def recording_and_plotting(dataset_name, name, alter, train, validation,
                           x_title="Sample size", y_title="Score"):
    train_scores_mean = np.mean(train, axis=1)
    train_scores_std = np.std(train, axis=1)
    test_scores_mean = np.mean(validation, axis=1)
    test_scores_std = np.std(validation, axis=1)

    # recording
    DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    DT_1.write('{}/{}'.format(dataset_name, name))
    DT_1.write("\n\n")
    DT_1.write(str(alter))
    DT_1.write("\n\n")
    DT_1.write(str(train_scores_mean))
    DT_1.write("\n\n")
    DT_1.write(str(test_scores_mean))

    # plotting
    plt.grid()
    ylim = (0, 1.1)
    plt.ylim(*ylim)
    plt.fill_between(alter, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(alter, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(alter, train_scores_mean, color="r",
             label="Training score")
    plt.plot(alter, test_scores_mean, color="g",
             label="Cross-validation score")
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('{}/{}.png'.format(dataset_name, name))
    plt.gcf().clear()
def ann_learning_curve_size_post(dataset_name, X_train, y_train, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
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
    print("ann_learning_curve_size_post", difference)

    recording_and_plotting(dataset_name, name="ann_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
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

def ann_learning_curve_size_RHC(dataset_name, X_train, y_train, hidden_nodes = [100], max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    # clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    clf = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                               algorithm='random_hill_climb', max_iters=max_iter,
                               bias=True, is_classifier=True, learning_rate=alpha,
                               early_stopping=True, clip_max=5, max_attempts=100,
                               restarts=20,
                               random_state=1)
    start_time = time.time()
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print("ann_learning_curve_size_post", difference)

    recording_and_plotting(dataset_name, name="ann_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")

def ann_RHC_vld_curve_1(dataset_name, X_train, y_train, hidden_nodes = [50], max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                               algorithm='random_hill_climb', max_iters=max_iter,
                               bias=True, is_classifier=True, learning_rate=alpha,
                               early_stopping=True, clip_max=5, max_attempts=100,
                               restarts=20,
                               random_state=1)
    start_time = time.time()
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cv = None
    param_range = []
    alter_list = []
    for i in range(1,120,20):
        param_range.append((i,))
        alter_list.append(i)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="hidden_layer_sizes",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print("ann_vld_curve_1", difference)

    recording_and_plotting(dataset_name, name="ann_RHC_vld_curve_1",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="hidden_layer_sizes", y_title="Score")
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

    # X_train = train[:, 1:]
    # y_train = train[:, 0]
    # X_test = test[:, 1:]
    # y_test = test[:, 0]

    X = train[:, 1:]
    y = train[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        shuffle=True,
                                                        stratify=y)
    print(y_train)
    # standardize the original data - this is important but usually neglected by newbies.
    scaler = preprocessing.StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
    set1_name = "mnist"

    # ann_learning_curve_size_pre(set1_name, X_train_scaled, y_train_hot)
    # ann_learning_curve_size_post(set1_name, X_train_scaled, y_train_hot, hidden_layer_sizes=(50,), alpha=6.25)

    # ann_learning_curve_size_RHC(set1_name, X_train_scaled, y_train_hot, hidden_nodes=[50], max_iter=500, alpha=0.0001)

    for i in range(10):
        start = time.time()
        print("===========i===========", i)
        clf = mlrose.NeuralNetwork(hidden_nodes=[50,], activation='relu',
                                   algorithm='random_hill_climb', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=i+1,
                                   # early_stopping=True,
                                   # clip_max=5,
                                   max_attempts=1000,
                                   # restarts=20,
                                   # schedule= mlrose.GeomDecay(),
                                   # pop_size = 200,
                                   # mutation_prob = 0.1,
                                   random_state=1)


        clf.fit(X_train_scaled, y_train_hot)
        # Predict labels for train set and assess accuracy
        y_train_pred = clf.predict(X_train_scaled)
        print("time:", time.time()-start)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        print(y_train_accuracy)

        #
        # # Predict labels for test set and assess accuracy
        # y_test_pred = clf.predict(X_test_scaled)
        # y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        # print(y_test_accuracy)
        #



