import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit, cross_val_score, cross_validate

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm


''' 
From Office Hour 1:
Each of the agrithm have several hyperparameters, for this assignment.  Grid search is sth ppl can do, in their live. But in this secific assignment, 
we want u to pick at least one, usually 2, hyperparameter of ur problem, and actually do the model complexity anaylsis for it. which means u actually change 
the values of these hyperparameters, plot the curve for training set, and also validation curve,  and then u compare u find ur best value for that HP.We
dont need u to do that for all the HP, for something like neuron analysis, that could be many, but of course if u want to learn the best, you know, model, u will have to do that anyways. 
But in ur analysis as long as u  provide the chart and  analysis for 2 of the HP, that would be good enough . 

For the rest of the them u can say w/e u did if u did , u know, model complexsity analysis,but u dont have to provide the chart, u just say that the result was, 
but we need to see the the some sort of the model complexity analysis for two of the HP parameters. '''


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

def decision_tree_learning_curve_size_pre(dataset_name, X_train, y_train, min_samples_leaf=25, max_depth=None):
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes,
                                                            )
    end_time = time.time()
    difference = end_time - start_time
    print "DT_learning_curve_size_pre", difference
    recording_and_plotting(dataset_name, name="DT_learning_curve_size_pre",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def boost_dt_learning_curve_size_pre(dataset_name, X_train, y_train, min_samples_leaf=25, n_estimators=5, learning_rate = 1.0):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf), n_estimators=n_estimators, learning_rate=learning_rate)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "boost_dt_learning_curve_size_pre", difference

    recording_and_plotting(dataset_name, name="boost_dt_learning_curve_size_pre",
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
    print "ann_learning_curve_size_pre", difference

    recording_and_plotting(dataset_name, name="ann_learning_curve_size_pre",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def knn_learning_curve_size_pre(dataset_name, X_train, y_train, n_neighbors=5, algorithm='auto'):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "knn_learning_curve_size_pre", difference

    recording_and_plotting(dataset_name, name="knn_learning_curve_size_pre",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def svm_learning_curve_size_pre(dataset_name, X_train, y_train, C=1.0, kernel='rbf', max_iter=-1):
    clf = svm.SVC(C=C, kernel=kernel,  max_iter=max_iter)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes
                                                            )
    end_time = time.time()
    difference = end_time - start_time
    print "svm_learning_curve_size_pre", difference

    recording_and_plotting(dataset_name, name="svm_learning_curve_size_pre",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")

def boost_dt_learning_curve_epoch_pre(dataset_name, X_train, y_train, min_samples_leaf=25, n_estimators=5, learning_rate = 1.0):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf), n_estimators=n_estimators, learning_rate=learning_rate)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 100, 4)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="n_estimators",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "boost_dt_learning_curve_epoch_pre", difference

    recording_and_plotting(dataset_name, name="boost_dt_learning_curve_epoch_pre",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_estimators", y_title="Score")
def ann_learning_curve_epoch_pre(dataset_name, X, y, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    start_time = time.time()
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    scores_train = []
    scores_test = []
    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_test = X[validation_index]
        y_train = y[train_index]
        y_test = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        """ Home-made mini-batch learning
            -> not to be used in out-of-core setting!
        """

        N_TRAIN_SAMPLES = X_train.shape[0]
        N_EPOCHS = 25
        N_BATCH = 128
        N_CLASSES = np.unique(y_train)

        scores_train_looper = []
        scores_test_looper = []
        n_alter_looper = []
        # EPOCH
        epoch = 0
        while epoch < N_EPOCHS:
            # print('epoch: ', epoch)
            # SHUFFLING
            random_perm = np.random.permutation(X_train.shape[0])
            mini_batch_index = 0
            while True:
                # MINI-BATCH
                indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
                clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
                mini_batch_index += N_BATCH

                if mini_batch_index >= N_TRAIN_SAMPLES:
                    break

            # SCORE TRAIN
            scores_train_looper.append(clf.score(X_train, y_train))

            # SCORE TEST
            scores_test_looper.append(clf.score(X_test, y_test))

            n_alter_looper.append(epoch)

            epoch += 1
            # print scores_train_looper
        scores_train.append(scores_train_looper)
        scores_test.append(scores_test_looper)
        # n_alter.append(n_alter_looper)
    end_time = time.time()
    difference = end_time - start_time
    print "ann_learning_curve_epoch_pre", difference
    n_alter = range(25)

    # recording
    name = 'ann_learning_curve_epoch_pre'
    # n_alter_mean = np.mean(n_alter, axis=1)
    train_scores_mean = np.mean(scores_train, axis=0)
    train_scores_std = np.std(scores_train, axis=0)
    test_scores_mean = np.mean(scores_test, axis=0)
    test_scores_std = np.std(scores_test, axis=0)
    DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    DT_1.write('{}/{}'.format(dataset_name, name))
    DT_1.write("\n\n")
    DT_1.write(str(n_alter))
    DT_1.write("\n\n")
    DT_1.write(str(train_scores_mean))
    DT_1.write("\n\n")
    DT_1.write(str(test_scores_mean))

    # plotting
    plt.grid()
    ylim = (0, 1.1)
    plt.ylim(*ylim)
    plt.fill_between(n_alter, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(n_alter, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(n_alter, train_scores_mean, color="r",
             label="Training score")
    plt.plot(n_alter, test_scores_mean, color="g",
             label="Cross-validation score")
    x_title = "Epochs"
    y_title = "Score"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('{}/{}.png'.format(dataset_name, name))
    # plt.savefig('22222.png')
    plt.gcf().clear()
def svm_learning_curve_epoch_pre(dataset_name, X_train, y_train, C=1.0, kernel='rbf', max_iter=-1):
    clf = svm.SVC(C=C, kernel=kernel,  max_iter=max_iter)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(0, 100, 4)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="max_iter",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "SVM epoch difference:", difference

    recording_and_plotting(dataset_name, name="SVM_learning_curve_epoch",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="max_iter", y_title="Score")

def decision_tree_learning_curve_size_post(dataset_name, X_train, y_train, min_samples_leaf=25, max_depth=None):
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes,
                                                            )
    end_time = time.time()
    difference = end_time - start_time
    print "DT_learning_curve_size_post", difference
    recording_and_plotting(dataset_name, name="DT_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def boost_dt_learning_curve_size_post(dataset_name, X_train, y_train, min_samples_leaf=25, n_estimators=5, learning_rate = 1.0):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf), n_estimators=n_estimators, learning_rate=learning_rate)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "boost_dt_learning_curve_size_post", difference

    recording_and_plotting(dataset_name, name="boost_dt_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
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
    print "ann_learning_curve_size_post", difference

    recording_and_plotting(dataset_name, name="ann_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def knn_learning_curve_size_post(dataset_name, X_train, y_train, n_neighbors=5, algorithm='auto'):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "knn_learning_curve_size_post", difference

    recording_and_plotting(dataset_name, name="knn_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")
def svm_learning_curve_size_post(dataset_name, X_train, y_train, C=1.0, kernel='rbf', max_iter=-1):
    clf = svm.SVC(C=C, kernel=kernel,  max_iter=max_iter)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes
                                                            )
    end_time = time.time()
    difference = end_time - start_time
    print "svm_learning_curve_size_post", difference

    recording_and_plotting(dataset_name, name="svm_learning_curve_size_post",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")

def boost_dt_learning_curve_epoch_post(dataset_name, X_train, y_train, min_samples_leaf=25, n_estimators=5, learning_rate = 1.0):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf), n_estimators=n_estimators, learning_rate=learning_rate)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 100, 4)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="n_estimators",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "boost_dt_learning_curve_epoch_post", difference

    recording_and_plotting(dataset_name, name="boost_dt_learning_curve_epoch_post",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_estimators", y_title="Score")
def ann_learning_curve_epoch_post(dataset_name, X, y, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    start_time = time.time()
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    scores_train = []
    scores_test = []
    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_test = X[validation_index]
        y_train = y[train_index]
        y_test = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        """ Home-made mini-batch learning
            -> not to be used in out-of-core setting!
        """

        N_TRAIN_SAMPLES = X_train.shape[0]
        N_EPOCHS = 25
        N_BATCH = 128
        N_CLASSES = np.unique(y_train)

        scores_train_looper = []
        scores_test_looper = []
        n_alter_looper = []
        # EPOCH
        epoch = 0
        while epoch < N_EPOCHS:
            # print('epoch: ', epoch)
            # SHUFFLING
            random_perm = np.random.permutation(X_train.shape[0])
            mini_batch_index = 0
            while True:
                # MINI-BATCH
                indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
                clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
                mini_batch_index += N_BATCH

                if mini_batch_index >= N_TRAIN_SAMPLES:
                    break

            # SCORE TRAIN
            scores_train_looper.append(clf.score(X_train, y_train))

            # SCORE TEST
            scores_test_looper.append(clf.score(X_test, y_test))

            n_alter_looper.append(epoch)

            epoch += 1
            # print scores_train_looper
        scores_train.append(scores_train_looper)
        scores_test.append(scores_test_looper)
        # n_alter.append(n_alter_looper)
    end_time = time.time()
    difference = end_time - start_time
    print "ann_learning_curve_epoch_post", difference
    n_alter = range(25)

    # recording
    name = 'ann_learning_curve_epoch_post'
    # n_alter_mean = np.mean(n_alter, axis=1)
    train_scores_mean = np.mean(scores_train, axis=0)
    train_scores_std = np.std(scores_train, axis=0)
    test_scores_mean = np.mean(scores_test, axis=0)
    test_scores_std = np.std(scores_test, axis=0)
    DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    DT_1.write('{}/{}'.format(dataset_name, name))
    DT_1.write("\n\n")
    DT_1.write(str(n_alter))
    DT_1.write("\n\n")
    DT_1.write(str(train_scores_mean))
    DT_1.write("\n\n")
    DT_1.write(str(test_scores_mean))

    # plotting
    plt.grid()
    ylim = (0, 1.1)
    plt.ylim(*ylim)
    plt.fill_between(n_alter, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(n_alter, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(n_alter, train_scores_mean, color="r",
             label="Training score")
    plt.plot(n_alter, test_scores_mean, color="g",
             label="Cross-validation score")
    x_title = "Epochs"
    y_title = "Score"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('{}/{}.png'.format(dataset_name, name))
    # plt.savefig('22222.png')
    plt.gcf().clear()
def svm_learning_curve_epoch_post(dataset_name, X_train, y_train, C=1.0, kernel='rbf', max_iter=-1):
    clf = svm.SVC(C=C, kernel=kernel,  max_iter=max_iter)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(0, 100, 4)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="max_iter",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "SVM epoch difference:", difference

    recording_and_plotting(dataset_name, name="SVM_learning_curve_epoch",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="max_iter", y_title="Score")

def decision_tree_vld_curve_1(dataset_name, X_train, y_train, min_samples_leaf=25, max_depth=None):
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = 10
    param_range = range(1, 200, 8)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="min_samples_leaf",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "DT_vlad_curve_1", difference

    recording_and_plotting(dataset_name, name="DT_vlad_curve_1",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="min_samples_leaf", y_title="Score")
def decision_tree_vld_curve_2(dataset_name, X_train, y_train, min_samples_leaf=25, max_depth=None):
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 50, 2)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="max_depth",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "DT_vlad_curve_2", difference

    recording_and_plotting(dataset_name, name="DT_vlad_curve_2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="max_depth", y_title="Score")

def boost_dt_vld_curve_1(dataset_name, X_train, y_train, min_samples_leaf=25, n_estimators=5, learning_rate = 1.0):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf), n_estimators=n_estimators, learning_rate=learning_rate)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = []
    alter_list = []
    for i in range(1, 200, 8):
        param_range.append(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=i))
        alter_list.append(i)
    # print param_range
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="base_estimator",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "boost_dt_vld_curve_1", difference

    recording_and_plotting(dataset_name, name="boost_dt_vld_curve_1",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="min_samples_leaf", y_title="Score")
def boost_dt_vld_curve_2(dataset_name, X_train, y_train, min_samples_leaf=25, n_estimators=5, learning_rate = 1.0):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf), n_estimators=n_estimators, learning_rate=learning_rate)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = np.linspace(0.01, 1, 25)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="learning_rate",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "BoostDT4 difference:", difference

    recording_and_plotting(dataset_name, name="BoostDT4",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="learning_rate", y_title="Score")

def ann_vld_curve_1(dataset_name, X_train, y_train, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = []
    alter_list = []
    for i in range(1,400,20):
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
    print "ann_vld_curve_1", difference

    recording_and_plotting(dataset_name, name="ann_vld_curve_1",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="hidden_layer_sizes", y_title="Score")
def ann_vld_curve_2(dataset_name, X_train, y_train, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = 5
    param_range = np.linspace(0.00001, 50, 25)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="alpha",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "ann_vld_curve_2", difference

    recording_and_plotting(dataset_name, name="ann_vld_curve_2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="alpha", y_title="Score")

def knn_vld_curve_1(dataset_name, X_train, y_train, n_neighbors=5, algorithm='auto'):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1,50,2)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="n_neighbors",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "knn_vld_curve_1", difference

    recording_and_plotting(dataset_name, name="knn_vld_curve_1",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_neighbors", y_title="Score")
def knn_vld_curve_2(dataset_name, X_train, y_train, n_neighbors=5, algorithm='auto'):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = ['auto', 'ball_tree', 'kd_tree', 'brute']
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="algorithm",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "knn_vld_curve_2", difference

    recording_and_plotting(dataset_name, name="knn_vld_curve_2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="algorithm", y_title="Score")

def svm_vld_curve_1(dataset_name, X_train, y_train, C=1.0, kernel='rbf', max_iter=-1):
    clf = svm.SVC(C=C, kernel=kernel,  max_iter=max_iter)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = np.linspace(0.01, 100000, 25)

    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="C",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "svm_vld_curve_1", difference

    recording_and_plotting(dataset_name, name="svm_vld_curve_1",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="C", y_title="Score")
def svm_vld_curve_2(dataset_name, X_train, y_train, C=1.0, kernel='rbf', max_iter=-1):
    clf = svm.SVC(C=C, kernel=kernel,  max_iter=max_iter)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = ['linear', 'poly', 'rbf', 'sigmoid']
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="kernel",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "svm_vld_curve_2", difference

    recording_and_plotting(dataset_name, name="svm_vld_curve_2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="kernel", y_title="Score")

if __name__=="__main__":
    '''load and standardize data set #1'''

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

    # # pre-parameter adjustment
    # decision_tree_learning_curve_size_pre(set1_name, X_train, y_train)
    # boost_dt_learning_curve_size_pre(set1_name, X_train, y_train)
    # ann_learning_curve_size_pre(set1_name, X_train, y_train)
    # knn_learning_curve_size_pre(set1_name, X_train, y_train)
    # svm_learning_curve_size_pre(set1_name, X_train, y_train)
    #
    # boost_dt_learning_curve_epoch_pre(set1_name, X_train, y_train)
    # ann_learning_curve_epoch_pre(set1_name, X_train, y_train)
    # svm_learning_curve_epoch_pre(set1_name, X_train, y_train)

    # parameter validation curve
    # decision_tree_vld_curve_1(set1_name, X_train, y_train)
    # decision_tree_vld_curve_2(set1_name, X_train, y_train)
    # boost_dt_vld_curve_1(set1_name, X_train, y_train)
    # boost_dt_vld_curve_2(set1_name, X_train, y_train)
    # ann_vld_curve_1(set1_name, X_train, y_train)
    # ann_vld_curve_2(set1_name, X_train, y_train)
    # knn_vld_curve_1(set1_name, X_train, y_train)
    # knn_vld_curve_2(set1_name, X_train, y_train)
    # svm_vld_curve_1(set1_name, X_train, y_train)
    # svm_vld_curve_2(set1_name, X_train,  y_train)

    # # post-parameter adjustment
    decision_tree_learning_curve_size_post(set1_name, X_train, y_train, min_samples_leaf=1, max_depth=None)
    boost_dt_learning_curve_size_post(set1_name, X_train, y_train, min_samples_leaf=9, n_estimators=40, learning_rate = 0.0925)
    ann_learning_curve_size_post(set1_name, X_train, y_train, hidden_layer_sizes=(50, ), alpha=6.25)
    knn_learning_curve_size_post(set1_name, X_train, y_train, n_neighbors=5, algorithm='auto')
    svm_learning_curve_size_post(set1_name, X_train, y_train, C=0.418, kernel='rbf', max_iter=-1)

    boost_dt_learning_curve_epoch_post(set1_name, X_train, y_train, min_samples_leaf=9, n_estimators=40, learning_rate = 0.0925)
    ann_learning_curve_epoch_post(set1_name, X_train, y_train, hidden_layer_sizes=(50, ), alpha=6.25)
    svm_learning_curve_epoch_post(set1_name, X_train, y_train, C=0.418, kernel='rbf', max_iter=-1)

    '''===========for seizure========='''
    set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, :]
    set2 = set2.astype(int)

    # separating set2 into X and y, then train and test
    X2 = set2[:, :-1]
    scaler = preprocessing.StandardScaler()
    X2 = scaler.fit_transform(X2)
    y2 = set2[:, -1]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    #
    set2_name = "seizure_5"

    # # pre-parameter adjustment
    # decision_tree_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # boost_dt_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # ann_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # knn_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # svm_learning_curve_size_pre(set2_name, X2_train, y2_train)
    #
    # boost_dt_learning_curve_epoch_pre(set2_name, X2_train, y2_train)
    # ann_learning_curve_epoch_pre(set2_name, X2_train, y2_train)
    # svm_learning_curve_epoch_pre(set2_name, X2_train, y2_train)

    # parameter validation curve
    # decision_tree_vld_curve_1(set2_name, X2_train, y2_train)
    # decision_tree_vld_curve_2(set2_name, X2_train, y2_train)
    # boost_dt_vld_curve_1(set2_name, X2_train, y2_train)
    # boost_dt_vld_curve_2(set2_name, X2_train, y2_train)
    # ann_vld_curve_1(set2_name, X2_train, y2_train)
    # ann_vld_curve_2(set2_name, X2_train, y2_train)
    # knn_vld_curve_1(set2_name, X2_train, y2_train)
    # knn_vld_curve_2(set2_name, X2_train, y2_train)
    svm_vld_curve_1(set2_name, X2_train, y2_train)
    # svm_vld_curve_2(set2_name, X2_train, y2_train)

    # # post-parameter adjustment
    # decision_tree_learning_curve_size_post(set2_name, X2_train, y2_train, min_samples_leaf=33, max_depth=None)
    # boost_dt_learning_curve_size_post(set2_name, X2_train, y2_train, min_samples_leaf=113, n_estimators=40, learning_rate = 0.5125)
    # ann_learning_curve_size_post(set2_name, X2_train, y2_train, hidden_layer_sizes=(200, ), alpha=0.417)
    # knn_learning_curve_size_post(set2_name, X2_train, y2_train, n_neighbors=1, algorithm='auto')
    # svm_learning_curve_size_post(set2_name, X2_train, y2_train, C= , kernel='rbf', max_iter=-1)
    #
    # boost_dt_learning_curve_epoch_post(set2_name, X2_train, y2_train, min_samples_leaf=113, n_estimators=40, learning_rate = 0.5125)
    # ann_learning_curve_epoch_post(set2_name, X2_train, y2_train, hidden_layer_sizes=(200, ), alpha=0.417)
    # svm_learning_curve_epoch_post(set2_name, X2_train, y2_train, C= , kernel='rbf', max_iter=-1)


    '''load and standardize data set #2'''

    # set2 = np.genfromtxt('bank-full.csv', delimiter=';', dtype=None)[1:, :]
    #
    # set2[set2 == '"unknown"'] = 0
    # set2[set2 == '"admin."'] = 1
    # set2[set2 == '"unemployed"'] = 2
    # set2[set2 == '"management"'] = 3
    # set2[set2 == '"housemaid"'] = 4
    # set2[set2 == '"entrepreneur"'] = 5
    # set2[set2 == '"student"'] = 6
    # set2[set2 == '"blue-collar"'] = 7
    # set2[set2 == '"self-employed"'] = 8
    # set2[set2 == '"retired"'] = 9
    # set2[set2 == '"technician"'] = 10
    # set2[set2 == '"services"'] = 11
    #
    # set2[set2 == '"married"'] = 0
    # set2[set2 == '"divorced"'] = 1
    # set2[set2 == '"single"'] = 2
    #
    # set2[set2 == '"jan"'] = 0
    # set2[set2 == '"feb"'] = 1
    # set2[set2 == '"mar"'] = 2
    # set2[set2 == '"apr"'] = 3
    # set2[set2 == '"may"'] = 4
    # set2[set2 == '"jun"'] = 5
    # set2[set2 == '"jul"'] = 6
    # set2[set2 == '"aug"'] = 7
    # set2[set2 == '"sep"'] = 8
    # set2[set2 == '"oct"'] = 9
    # set2[set2 == '"nov"'] = 10
    # set2[set2 == '"dec"'] = 11
    #
    # set2[set2 == '"yes"'] = 1
    # set2[set2 == '"no"'] = 0
    #
    # set2[set2 == '"secondary"'] = 1
    # set2[set2 == '"primary"'] = 2
    # set2[set2 == '"tertiary"'] = 3
    #
    # set2[set2 == '"telephone"'] = 1
    # set2[set2 == '"cellular"'] = 2
    #
    # set2[set2 == '"other"'] = 1
    # set2[set2 == '"failure"'] = 2
    # set2[set2 == '"success"'] = 3
    #
    # set2 = set2.astype(int)
    #
    # # separating set2 into X and y, then train and test
    # X2 = set2[:, :-1]
    # # print X2[:5, :]
    # scaler = preprocessing.StandardScaler()
    # X2 = scaler.fit_transform(X2)
    # # print X2[:5, :]
    # y2 = set2[:, -1]
    # # print y2
    # X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    # #
    # set2_name = "bank"
    #
    # # Decision tree experiment:
    # decision_tree_experiment_1(set2_name, X2_train, y2_train)
    # decision_tree_experiment_2(set2_name, X2_train, y2_train)  # Leaf size vs Accuracy  (Pruning)
    # decision_tree_experiment_3(set2_name, X2_train, y2_train)  # Max depth vs Accuracy
    #
    # # Boosted decision tree experiment:
    # boost_dt_experiment_1(set2_name, X2_train, y2_train)
    # boost_dt_experiment_2(set2_name, X2_train, y2_train)
    # boost_dt_experiment_3(set2_name, X2_train, y2_train)  # Leaf size vs Accuracy  (Pruning)
    # boost_dt_experiment_4(set2_name, X2_train, y2_train)
    #
    # # ANN experiment 1: Sample size vs Accuracy
    # ann_experiment_1(set2_name, X2_train, y2_train)
    # ann_experiment_2(set2_name, X2_train, y2_train)
    # ann_experiment_3(set2_name, X2_train, y2_train)
    #
    # # KNN experiment 1: Sample size vs Accuracy
    # knn_experiment_1(set2_name, X2_train, y2_train)
    # knn_experiment_2(set2_name, X2_train, y2_train)  # n_neighbours vs. score
    # knn_experiment_3(set2_name, X2_train, y2_train)  # algorithm vs. score
    #
    # # SVM experiment 1: Sample size vs Accuracy
    # svm_experiment_1(set2_name, X2_train, y2_train)
    # svm_experiment_2(set2_name, X2_train, y2_train)  # C vs. score
    # svm_experiment_3(set2_name, X2_train, y2_train)  # kernel vs. score
    '''==================for the loc================'''
    # train = np.genfromtxt('trainingData.csv', delimiter=',')[1:5000, :]
    # test = np.genfromtxt('validationData.csv', delimiter=',')[1:1000, :]
    # # train = np.genfromtxt('fashion-mnist_train.csv', delimiter=',')[1:, :]
    # # test = np.genfromtxt('fashion-mnist_test.csv', delimiter=',')[1:, :]
    #
    # X_train = train[:, :-9]
    # y_train = 10 * train[:, -6] + train[:, -7]
    # X_test = test[:, :-9]
    # y_test = 10 * test[:, -6] + test[:, -7]
    #
    # # standardize the original data - this is important but usually neglected by newbies.
    # scaler = preprocessing.StandardScaler()
    # # print X_train[:5, :]
    # X_train = scaler.fit_transform(X_train)
    # # print X_train[:5, :]
    # X_test = scaler.transform(X_test)
    #
    # set1_name = "loc"
    #
    # # Decision tree experiment 1: Sample size vs Accuracy
    # decision_tree_experiment_1(set1_name, X_train, y_train)
    # decision_tree_experiment_2(set1_name, X_train, y_train)  # Leaf size vs Accuracy  (Pruning)
    # decision_tree_experiment_3(set1_name, X_train, y_train)  # Max depth vs Accuracy
    #
    # # Boosted decision tree experiment 1: Sample size vs Accuracy
    # boost_dt_experiment_1(set1_name, X_train, y_train)
    # boost_dt_experiment_2(set1_name, X_train, y_train)
    # boost_dt_experiment_3(set1_name, X_train, y_train)
    # boost_dt_experiment_4(set1_name, X_train, y_train)
    #
    # # ANN experiment 1: Sample size vs Accuracy
    # ann_experiment_1(set1_name, X_train, y_train)
    # ann_experiment_2(set1_name, X_train, y_train)
    # ann_experiment_3(set1_name, X_train, y_train)
    #
    # # KNN experiment 1: Sample size vs Accuracy
    # knn_experiment_1(set1_name, X_train, y_train)
    # knn_experiment_2(set1_name, X_train, y_train)  # n_neighbours vs. score
    # knn_experiment_3(set1_name, X_train, y_train)  # algorithm vs. score
    #
    # # SVM experiment 1: Sample size vs Accuracy
    # svm_experiment_1(set1_name, X_train, y_train)
    # svm_experiment_3(set1_name, X_train, y_train)  # kernel vs. score
    # svm_experiment_2(set1_name, X_train, y_train)  # C vs. score








