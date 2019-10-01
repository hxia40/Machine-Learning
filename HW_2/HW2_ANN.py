import numpy as np
import pandas as pd
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
    print "SVM_learning_curve_epoch_post", difference

    recording_and_plotting(dataset_name, name="SVM_learning_curve_epoch_post",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="max_iter", y_title="Score")

def decision_tree_vld_curve_1(dataset_name, X_train, y_train, min_samples_leaf=25, max_depth=None):
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = 10
    param_range = range(1, 120, 5)
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
    param_range = np.linspace(0.01, 50, 25)

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

def score_time(dataset_name, clf_name, clf, X_train, X_test, y_train, y_test):
    start_time = time.time()
    score = 0
    difference = 0
    for i in range(10):
        clf.fit(X_train, y_train).predict(X_test)
        score += accuracy_score(y_test, clf.fit(X_train, y_train).predict(X_test))
        end_time = time.time()
        difference += (end_time - start_time)
    txt = open('Inter_model_comparison.txt', 'a')
    txt.write('{}_{} score:'.format(dataset_name, clf_name))
    txt.write(str(score/10))
    txt.write("\n")
    txt.write('{}_{} time:'.format(dataset_name, clf_name))
    txt.write(str(difference/10))
    txt.write("\n\n")
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

    '''MNIST - pre-parameter adjustment'''
    decision_tree_learning_curve_size_pre(set1_name, X_train, y_train)
    boost_dt_learning_curve_size_pre(set1_name, X_train, y_train)
    ann_learning_curve_size_pre(set1_name, X_train, y_train)
    knn_learning_curve_size_pre(set1_name, X_train, y_train)
    svm_learning_curve_size_pre(set1_name, X_train, y_train)

    boost_dt_learning_curve_epoch_pre(set1_name, X_train, y_train)
    ann_learning_curve_epoch_pre(set1_name, X_train, y_train)
    svm_learning_curve_epoch_pre(set1_name, X_train, y_train)

    '''MNIST - parameter validation curve'''
    decision_tree_vld_curve_1(set1_name, X_train, y_train)
    decision_tree_vld_curve_2(set1_name, X_train, y_train)
    boost_dt_vld_curve_1(set1_name, X_train, y_train)
    boost_dt_vld_curve_2(set1_name, X_train, y_train)
    ann_vld_curve_1(set1_name, X_train, y_train)
    ann_vld_curve_2(set1_name, X_train, y_train)
    knn_vld_curve_1(set1_name, X_train, y_train)
    knn_vld_curve_2(set1_name, X_train, y_train)
    svm_vld_curve_1(set1_name, X_train, y_train)
    svm_vld_curve_2(set1_name, X_train,  y_train)

    '''MNIST - post-parameter adjustment'''
    decision_tree_learning_curve_size_post(set1_name, X_train, y_train, min_samples_leaf=1, max_depth=None)
    boost_dt_learning_curve_size_post(set1_name, X_train, y_train, min_samples_leaf=9, n_estimators=40, learning_rate = 0.0925)
    ann_learning_curve_size_post(set1_name, X_train, y_train, hidden_layer_sizes=(50, ), alpha=6.25)
    knn_learning_curve_size_post(set1_name, X_train, y_train, n_neighbors=5, algorithm='auto')
    svm_learning_curve_size_post(set1_name, X_train, y_train, C=0.418, kernel='rbf', max_iter=-1)

    boost_dt_learning_curve_epoch_post(set1_name, X_train, y_train, min_samples_leaf=9, n_estimators=40, learning_rate = 0.0925)
    ann_learning_curve_epoch_post(set1_name, X_train, y_train, hidden_layer_sizes=(50, ), alpha=6.25)
    svm_learning_curve_epoch_post(set1_name, X_train, y_train, C=0.418, kernel='rbf', max_iter=-1)

    #

    '''Load and standardize data set ESR'''
    set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, :]
    set2 = set2.astype(int)

    # separating set2 into X and y, then train and test
    X2 = set2[:, :-1]
    scaler = preprocessing.StandardScaler()
    X2 = scaler.fit_transform(X2)
    y2 = set2[:, -1]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    #
    set2_name = "ESR"

    '''ESR - pre-parameter adjustment'''
    # decision_tree_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # boost_dt_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # ann_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # knn_learning_curve_size_pre(set2_name, X2_train, y2_train)
    # svm_learning_curve_size_pre(set2_name, X2_train, y2_train)

    # boost_dt_learning_curve_epoch_pre(set2_name, X2_train, y2_train)
    # ann_learning_curve_epoch_pre(set2_name, X2_train, y2_train)
    # svm_learning_curve_epoch_pre(set2_name, X2_train, y2_train)

    '''ESR - parameter validation curve'''
    decision_tree_vld_curve_1(set2_name, X2_train, y2_train)
    # decision_tree_vld_curve_2(set2_name, X2_train, y2_train)
    # boost_dt_vld_curve_1(set2_name, X2_train, y2_train)
    # boost_dt_vld_curve_2(set2_name, X2_train, y2_train)
    # ann_vld_curve_1(set2_name, X2_train, y2_train)
    # ann_vld_curve_2(set2_name, X2_train, y2_train)
    # knn_vld_curve_1(set2_name, X2_train, y2_train)
    # knn_vld_curve_2(set2_name, X2_train, y2_train)
    # svm_vld_curve_1(set2_name, X2_train, y2_train)
    # svm_vld_curve_2(set2_name, X2_train, y2_train)

    '''ESR - post-parameter adjustment'''
    # decision_tree_learning_curve_size_post(set2_name, X2_train, y2_train, min_samples_leaf=33, max_depth=None)
    # boost_dt_learning_curve_size_post(set2_name, X2_train, y2_train, min_samples_leaf=113, n_estimators=40, learning_rate = 0.5125)
    # ann_learning_curve_size_post(set2_name, X2_train, y2_train, hidden_layer_sizes=(50, ), alpha=0.417)
    # knn_learning_curve_size_post(set2_name, X2_train, y2_train, n_neighbors=1, algorithm='auto')
    # svm_learning_curve_size_post(set2_name, X2_train, y2_train, C=50 , kernel='rbf', max_iter=-1)
    # #
    # boost_dt_learning_curve_epoch_post(set2_name, X2_train, y2_train, min_samples_leaf=113, n_estimators=40, learning_rate = 0.5125)
    # ann_learning_curve_epoch_post(set2_name, X2_train, y2_train, hidden_layer_sizes=(50, ), alpha=0.417)
    # svm_learning_curve_epoch_post(set2_name, X2_train, y2_train, C=50 , kernel='rbf', max_iter=-1)

    # '''Inter-model comparison'''

    # open('Inter_model_comparison.txt', 'w')

    # score_time("MNIST", "Decision tree",
    #            tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, max_depth=None),
    #            X_train, X_test, y_train, y_test)
    # score_time("MNIST", "Boosting",
    #            AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=9),
    #                                   n_estimators=40, learning_rate=0.0925),
    #            X_train, X_test, y_train, y_test)
    # score_time("MNIST", "ANN",
    #            MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500, alpha=6.25, random_state=1),
    #            X_train, X_test, y_train, y_test)
    # score_time("MNIST", "kNN",
    #            neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto'),
    #            X_train, X_test, y_train, y_test)
    # score_time("MNIST", "SVM",
    #            svm.SVC(C=0.418, kernel='rbf', max_iter=-1),
    #            X_train, X_test, y_train, y_test)

    # score_time("ESR", "Decision tree",
    #            tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=33, max_depth=None),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time("ESR", "Boosting",
    #            AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=113),
    #                                   n_estimators=40, learning_rate=0.5125),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time("ESR", "ANN",
    #            MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500, alpha=0.417, random_state=1),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time("ESR", "kNN",
    #            neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='auto'),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time("ESR", "SVM",
    #            svm.SVC(C=50, kernel='rbf', max_iter=-1),
    #            X2_train, X2_test, y2_train, y2_test)


    # '''Inter-model comparison using default hyperparameters'''

    # open('Inter_model_comparison_default.txt', 'w')

    # score_time_default("MNIST", "Decision tree",
    #            tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None),
    #            X_train, X_test, y_train, y_test)
    # score_time_default("MNIST", "Boosting",
    #            AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25),
    #                                   n_estimators=5, learning_rate=1),
    #            X_train, X_test, y_train, y_test)
    # score_time_default("MNIST", "ANN",
    #            MLPClassifier(hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001, random_state=1),
    #            X_train, X_test, y_train, y_test)
    # score_time_default("MNIST", "kNN",
    #            neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto'),
    #            X_train, X_test, y_train, y_test)
    # score_time_default("MNIST", "SVM",
    #            svm.SVC(C=1, kernel='rbf', max_iter=-1),
    #            X_train, X_test, y_train, y_test)

    # score_time_default("ESR", "Decision tree",
    #            tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time_default("ESR", "Boosting",
    #            AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25),
    #                                   n_estimators=5, learning_rate=1),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time_default("ESR", "ANN",
    #            MLPClassifier(hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001, random_state=1),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time_default("ESR", "kNN",
    #            neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto'),
    #            X2_train, X2_test, y2_train, y2_test)
    # score_time_default("ESR", "SVM",
    #            svm.SVC(C=1, kernel='rbf', max_iter=-1),
    #            X2_train, X2_test, y2_train, y2_test)

    #





