import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit

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


def decision_tree_experiment_1(dataset_name, X_train, y_train): # Decision tree experiment 1: Sample size vs Accuracy
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None) #original
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
    print "DT difference:", difference
    print train_scores, type(train_scores)
    recording_and_plotting(dataset_name, name="DT1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def decision_tree_experiment_2(dataset_name, X_train, y_train): # Decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = 10
    param_range = range(1, 50, 2)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="min_samples_leaf",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "DT 2 difference:", difference

    recording_and_plotting(dataset_name, name="DT2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="Minimum leaf size", y_title="Score")


def decision_tree_experiment_3(dataset_name, X_train, y_train): # Decision tree experiment 3: Max depth vs Accuracy  (Pruning)
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 50, 2)
    print param_range
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="max_depth",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "DT 3 difference:", difference

    recording_and_plotting(dataset_name, name="DT3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="Max tree depth", y_title="Score")


def decision_tree_experiment_4(dataset_name, X_train, y_train): # Decision tree experiment 4: Sample size vs Accuracy
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, max_depth=None)  #opt
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "DT difference:", difference

    recording_and_plotting(dataset_name, name="DT4",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def boost_dt_experiment_1(dataset_name, X_train, y_train): # Boosted decision tree experiment 1: Sample size vs Accuracy
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25), n_estimators=5, learning_rate=1.0)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "BoostDT1 difference:", difference

    recording_and_plotting(dataset_name, name="BoostDT1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def boost_dt_experiment_2(dataset_name, X_train, y_train): # Boosted Decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25), n_estimators=5, learning_rate=1.0)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = []
    alter_list = []
    for i in range(1, 50, 2):
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
    print "BoostDT2 difference:", difference

    recording_and_plotting(dataset_name, name="BoostDT2",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="Minimum leaf size in Base estimator", y_title="Score")


def boost_dt_experiment_3(dataset_name, X_train, y_train): # Boosted Decision tree experiment 3: n_estimators vs Score
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25),
                             n_estimators=5,
                             learning_rate=1.0)
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
    print "BoostDT3 difference:", difference

    recording_and_plotting(dataset_name, name="BoostDT3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_estimators", y_title="Score")


def boost_dt_experiment_4(dataset_name, X_train, y_train): # Boosted Decision tree experiment 4: Learning rate vs Accuracy
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25),
                             n_estimators=5,
                             learning_rate=1.0)
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


def boost_dt_experiment_5(dataset_name, X_train, y_train): # Boosted decision tree experiment 1: Sample size vs Accuracy
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=2), n_estimators=50, learning_rate=0.07)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "BoostDT1 difference:", difference

    recording_and_plotting(dataset_name, name="BoostDT5",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def ann_experiment_1(dataset_name, X_train, y_train): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, max_iter=500, alpha=0.0001)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "ANN difference:", difference

    recording_and_plotting(dataset_name, name="ANN1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def ann_experiment_2(dataset_name, X_train, y_train): # ANN experiment 2: hidden layer size
    clf = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, alpha=0.0001)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = []
    alter_list = []
    for i in range(1,120,5):
        param_range.append((i,))
        alter_list.append(i)
    print param_range
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="hidden_layer_sizes",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "ANN2 difference:", difference

    recording_and_plotting(dataset_name, name="ANN2",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="Hidden layer size", y_title="Score")


def ann_experiment_3(dataset_name, X_train, y_train): # ANN experiment 3: alpha
    clf = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, alpha=0.0001)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = 5
    param_range = np.linspace(0.00001, 10, 25)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="alpha",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "ANN3 difference:", difference

    recording_and_plotting(dataset_name, name="ANN3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="alpha", y_title="Score")


def ann_experiment_4(dataset_name, X_train, y_train, X_test, y_test): # ANN experiment 3: alpha
    # mlp = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, alpha=0.0001)
    # from sklearn.datasets import fetch_mldata
    # np.random.seed(1)
    # mnist = fetch_mldata("MNIST original")
    # # rescale the data, use the traditional train/test split
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                        solver='adam', verbose=0, tol=1e-8, random_state=1,
                        learning_rate_init=.01)

    """ Home-made mini-batch learning
        -> not to be used in out-of-core setting!
    """

    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 25
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []
    n_alter = []
    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(mlp.score(X_test, y_test))

        n_alter.append(epoch)

        epoch += 1

    # recording
    dataset_name = 'seizure_5'
    name = 'ANN4'
    train_scores_mean = scores_train
    test_scores_mean = scores_test
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
    # plt.fill_between(alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(alter, train_scores_mean, color="r",
             label="Training score")
    plt.plot(alter, test_scores_mean, color="g",
             label="Cross-validation score")
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    # plt.savefig('seizure_5/ANN4.png')
    plt.savefig('22222.png')
    plt.gcf().clear()


def ann_experiment_5(dataset_name, X_train, y_train): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=(20, ), random_state=1, alpha=0.75)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "ANN difference:", difference

    recording_and_plotting(dataset_name, name="ANN5",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def knn_experiment_1(dataset_name, X_train, y_train): # ANN experiment 1: Sample size vs Accuracy
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm = 'auto')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "KNN difference:", difference

    recording_and_plotting(dataset_name, name="KNN1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def knn_experiment_2(dataset_name, X_train, y_train): # KNN experiment 2: n_neighbours vs. score
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1,50,2)
    print param_range
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="n_neighbors",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "KNN 2 difference:", difference

    recording_and_plotting(dataset_name, name="KNN2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_neighbors", y_title="Score")


def knn_experiment_3(dataset_name, X_train, y_train): # Decision tree experiment 3: p vs score
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm = 'auto')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = ['auto', 'ball_tree', 'kd_tree', 'brute']
    print param_range
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="algorithm",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "KNN 3 difference:", difference

    recording_and_plotting(dataset_name, name="KNN3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="algorithm", y_title="Score")


def knn_experiment_4(dataset_name, X_train, y_train): # ANN experiment 1: Sample size vs Accuracy
    clf = neighbors.KNeighborsClassifier(n_neighbors=8, algorithm='auto')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "KNN difference:", difference

    recording_and_plotting(dataset_name, name="KNN4",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def svm_experiment_1(dataset_name, X_train, y_train): # SVM experiment 1: Sample size vs Accuracy
    clf = svm.SVC(kernel='rbf')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    print train_sizes
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes
                                                            )
    end_time = time.time()
    difference = end_time - start_time
    print "SVM difference:", difference

    recording_and_plotting(dataset_name, name="SVM1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def svm_experiment_2(dataset_name, X_train, y_train): # SVM experiment 2: C
    clf = svm.SVC(C = 1.0, kernel='rbf')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = np.linspace(0.01, 100, 25)
    print param_range
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="C",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "SVM 2 difference:", difference

    recording_and_plotting(dataset_name, name="SVM2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="C", y_title="Score")


def svm_experiment_3(dataset_name, X_train, y_train): # SVM experiment 2: C alternation
    clf = svm.SVC(C = 1.0, kernel='rbf')
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
    print "SVM 3 difference:", difference

    recording_and_plotting(dataset_name, name="SVM3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="kernel", y_title="Score")


def svm_experiment_4(dataset_name, X_train, y_train): # SVM experiment 1: Sample size vs Accuracy
    clf = svm.SVC(kernel='rbf', C=4)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 25)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "SVM 4 difference:", difference

    recording_and_plotting(dataset_name, name="SVM4",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


if __name__=="__main__":
    '''load and standardize data set #1'''

    train = np.genfromtxt('fashion-mnist_train_minor.csv', delimiter=',')[1:, :]
    test = np.genfromtxt('fashion-mnist_test_minor.csv', delimiter=',')[1:, :]
    # train = np.genfromtxt('fashion-mnist_train.csv', delimiter=',')[1:, :]
    # test = np.genfromtxt('fashion-mnist_test.csv', delimiter=',')[1:, :]

    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_test = test[:, 1:]
    y_test = test[:, 0]

    # standardize the original data - this is important but usually neglected by newbies.
    scaler = preprocessing.StandardScaler()
    # print X_train[:5, :]
    X_train = scaler.fit_transform(X_train)
    # print X_train[:5, :]
    X_test = scaler.transform(X_test)

    set1_name = "mnist"

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

    # SVM experiment 1: Sample size vs Accuracy
    # svm_experiment_1(set1_name, X_train, y_train)
    # svm_experiment_3(set1_name, X_train, y_train)  # kernel vs. score
    # svm_experiment_2(set1_name, X_train, y_train)  # C vs. score

    # # Post-optimization learning curve
    # decision_tree_experiment_4(set1_name, X_train, y_train)
    # ann_experiment_4(set1_name, X_train, y_train)
    # knn_experiment_4(set1_name, X_train, y_train)
    # boost_dt_experiment_5(set1_name, X_train, y_train)
    # svm_experiment_4(set1_name, X_train, y_train)

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

    '''===========for seizure========='''
    # set2 = np.genfromtxt('Epileptic_Seizure_Recognition_binary.csv', delimiter=',', dtype=None)[1:1001, :]
    set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, :]

    set2 = set2.astype(int)

    # separating set2 into X and y, then train and test
    X2 = set2[:, :-1]
    # print X2[:5, :]
    scaler = preprocessing.StandardScaler()
    X2 = scaler.fit_transform(X2)
    # print X2[:5, :]
    y2 = set2[:, -1]
    # print y2
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    #
    set2_name = "seizure_5"

    # # # Decision tree experiment:
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
    # # ann_experiment_4(set2_name, X2_train, y2_train, X2_test, y2_test)
    #
    # # KNN experiment 1: Sample size vs Accuracy
    # knn_experiment_1(set2_name, X2_train, y2_train)
    # knn_experiment_2(set2_name, X2_train, y2_train)  # n_neighbours vs. score
    # knn_experiment_3(set2_name, X2_train, y2_train)  # algorithm vs. score

    # SVM experiment 1: Sample size vs Accuracy
    # svm_experiment_1(set2_name, X2_train, y2_train)
    svm_experiment_2(set2_name, X2_train, y2_train)  # C vs. score
    # svm_experiment_3(set2_name, X2_train, y2_train)  # kernel vs. score



