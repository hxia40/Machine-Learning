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


'''Each of the agrithm have several hyperparameters, for this assignment.  Grid search is sth ppl can do, in their live. But in this secific assignment, 
we want u to pick at least one, usually 2, hyperparameter of ur problem, and actually do the model complexity anaylsis for it. which means u actually change 
the values of these hyperparameters, plot the curve for training set, and also validation curve,  and then u compare u find ur best value for that HP.We
dont need u to do that for all the HP, for something like neuron analysis, that could be many, but of course if u want to learn the best, you know, model, u will have to do that anyways. 
But in ur analysis as long as u  provide the chart and  analysis for 2 of the HP, that would be good enough . 

For the rest of the them u can say w/e u did if u did , u know, model complexsity analysis,but u dont have to provide the chart, u just say that the result was, 
but we need to see the the some sort of the model complexity analys for two of the HP parameters. '''


def recording_and_plotting(name, alter, train, validation,
                           x_title="Sample size", y_title="Score"):
    # recording
    DT_1 = open('{}.txt'.format(name), 'w')
    DT_1.write(str(alter))
    DT_1.write("\n\n")
    DT_1.write(str(train))
    DT_1.write("\n\n")
    DT_1.write(str(validation))

    # plotting
    train_scores_mean = np.mean(train, axis=1)
    train_scores_std = np.std(train, axis=1)
    test_scores_mean = np.mean(validation, axis=1)
    test_scores_std = np.std(validation, axis=1)
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
    plt.savefig('{}.png'.format(name))
    plt.gcf().clear()


def decision_tree_model(X_tr, y_tr, X_te, y_te, min_samples_leaf):
    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    # print X_tr
    # print y_tr
    # choose decision tree classifier
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf)

    # cvs = cross_val_score(clf, X_tr, y_tr, cv=10, scoring='accuracy')
    # print cvs
    clf.fit(X_tr, y_tr)

    # use metrics to evaluate how good the performance is
    y_tr_predict = clf.predict(X_tr)

    # Internal test, using training set for accurate performance evaluation
    internal_score = metrics.accuracy_score(y_tr, y_tr_predict)

    # Standard external test, using test set for accurate performance evaluation
    y_predict = clf.predict(X_te)
    external_score = metrics.accuracy_score(y_te, y_predict)

    return internal_score, external_score
    # return 1,2


def boost_dt_model(X_tr, y_tr, X_te, y_te, min_samples_leaf, n_estimators):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import tree
    # choose AdaBoostClassifier classifier
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf)
    boost = AdaBoostClassifier(base_estimator=clf, n_estimators=n_estimators, learning_rate=1.0, algorithm='SAMME.R',
                                random_state=None)
    boost.fit(X_tr, y_tr)

    # use metrics to evaluate how good the performance is
    y_tr_predict = boost.predict(X_tr)

    # Internal test, using training set for accurate performance evaluation
    internal_score = metrics.accuracy_score(y_tr, y_tr_predict)

    # Standard external test, using test set for accurate performance evaluation
    y_predict = boost.predict(X_te)
    external_score = metrics.accuracy_score(y_te, y_predict)

    return internal_score, external_score


def ann_model(X_tr, y_tr, X_te, y_te, hidden_layer_sizes=(5, )):
    from sklearn.neural_network import MLPClassifier
    # choose MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=500)
    clf.fit(X_tr, y_tr)

    # use metrics to evaluate how good the performance is
    y_tr_predict = clf.predict(X_tr)

    # Internal test, using training set for accurate performance evaluation
    internal_score = metrics.accuracy_score(y_tr, y_tr_predict)

    # Standard external test, using test set for accurate performance evaluation
    y_predict = clf.predict(X_te)
    external_score = metrics.accuracy_score(y_te, y_predict)

    return internal_score, external_score


def knn_model(X_tr, y_tr, X_te, y_te, n_neighbors=5):
    from sklearn import neighbors
    # choose KNeighbors classifier
    clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_tr, y_tr)

    # use metrics to evaluate how good the performance is
    y_tr_predict = clf.predict(X_tr)

    # Internal test, using training set for accurate performance evaluation
    internal_score = metrics.accuracy_score(y_tr, y_tr_predict)

    # Standard external test, using test set for accurate performance evaluation
    y_predict = clf.predict(X_te)
    external_score = metrics.accuracy_score(y_te, y_predict)

    return internal_score, external_score


def svm_model(X_tr, y_tr, X_te, y_te, kernel='rbf'):
    from sklearn import svm
    # choose KNeighbors classifier
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_tr, y_tr)

    # use metrics to evaluate how good the performance is
    y_tr_predict = clf.predict(X_tr)

    # Internal test, using training set for accurate performance evaluation
    internal_score = metrics.accuracy_score(y_tr, y_tr_predict)

    # Standard external test, using test set for accurate performance evaluation
    y_predict = clf.predict(X_te)
    external_score = metrics.accuracy_score(y_te, y_predict)

    return internal_score, external_score


def decision_tree_experiment_1(X_train, y_train): # Decision tree experiment 1: Sample size vs Accuracy
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    train_sizes = np.linspace(.01, 1.0, 50)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv = cv,
                                                            train_sizes = train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "DT difference:", difference

    recording_and_plotting(name="DT1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def decision_tree_experiment_2(X_train, y_train): # Decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 40, 2)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="min_samples_leaf",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "DT 2 difference:", difference

    recording_and_plotting(name="DT2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="Minimum leaf size", y_title="Score")


def decision_tree_experiment_3(X_train, y_train): # Decision tree experiment 3: Max depth vs Accuracy  (Pruning)
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25, max_depth=None)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 40, 2)
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

    recording_and_plotting(name="DT3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="Max tree depth", y_title="Score")


def boost_dt_experiment_1(X_train, y_train): # Boosted decision tree experiment 1: Sample size vs Accuracy
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25), n_estimators=5, learning_rate=1.0)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 50)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "BoostDT1 difference:", difference

    recording_and_plotting(name="BoostDT1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def boost_dt_experiment_2(X_train, y_train): # Boosted Decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25), n_estimators=5, learning_rate=1.0)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = []
    alter_list = []
    for i in range(1, 40, 2):
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

    recording_and_plotting(name="BoostDT2",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="Minimum leaf size in Base estimator", y_title="Score")


def boost_dt_experiment_3(X_train, y_train): # Boosted Decision tree experiment 3: Number of estim vs Accuracy
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25),
                             n_estimators=5,
                             learning_rate=1.0)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 100, 2)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="n_estimators",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "BoostDT3 difference:", difference

    recording_and_plotting(name="BoostDT3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_estimators", y_title="Score")

def boost_dt_experiment_4(X_train, y_train): # Boosted Decision tree experiment 4: Learning rate vs Accuracy
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=25),
                             n_estimators=5,
                             learning_rate=1.0)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = np.linspace(0.01, 1, 50)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="learning_rate",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "BoostDT4 difference:", difference

    recording_and_plotting(name="BoostDT4",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="learning_rate", y_title="Score")


def ann_experiment_1(X_train, y_train): # ANN experiment 1: Sample size vs Accuracy
    clf = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, max_iter=500)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 50)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "ANN difference:", difference

    recording_and_plotting(name="ANN1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def ann_experiment_2(X_train, y_train): # ANN experiment 2: hidden layer size
    clf = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, max_iter=500)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = []
    alter_list = []
    for i in range(1,120,2):
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

    recording_and_plotting(name="ANN2",
                           alter=alter_list,
                           train=train_scores,
                           validation=test_scores, x_title="Hidden layer size", y_title="Score")

def ann_experiment_3(X_train, y_train): # ANN experiment 3: alpha
    clf = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, max_iter=500)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = np.linspace(0.01, 1, 50)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="alpha",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "ANN3 difference:", difference

    recording_and_plotting(name="ANN3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="alpha", y_title="Score")


def knn_experiment_1(X_train, y_train): # ANN experiment 1: Sample size vs Accuracy
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 50)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "KNN difference:", difference

    recording_and_plotting(name="KNN1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def knn_experiment_2(X_train, y_train): # Decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = range(1, 40, 2)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="n_neighbors",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print "KNN 2 difference:", difference

    recording_and_plotting(name="KNN2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="n_neighbors", y_title="Score")


def svm_experiment_1(X_train, y_train): # SVM experiment 1: Sample size vs Accuracy
    clf = svm.SVC(kernel='rbf')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 50)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train,
                                                            cv=cv,
                                                            train_sizes=train_sizes)
    end_time = time.time()
    difference = end_time - start_time
    print "SVM difference:", difference

    recording_and_plotting(name="SVM1",
                           alter=train_sizes,
                           train=train_scores,
                           validation=test_scores, x_title="Sample size", y_title="Score")


def svm_experiment_2(X_train, y_train): # SVM experiment 2: C
    clf = svm.SVC(C = 1.0, kernel='rbf')
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None
    param_range = np.linspace(0.01, 1.0, 50)
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

    recording_and_plotting(name="SVM2",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="C", y_title="Score")


def svm_experiment_3(X_train, y_train): # SVM experiment 2: C alternation
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

    recording_and_plotting(name="SVM3",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="kernel", y_title="Score")


if __name__=="__main__":
    # load and standardize data set #1

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # load and standardize data set #2
    # pass

    # # Decision tree experiment 1: Sample size vs Accuracy
    # decision_tree_experiment_1(X_train, y_train)
    #
    # # Decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    decision_tree_experiment_2(X_train, y_train) # Leaf size vs Accuracy  (Pruning)
    decision_tree_experiment_3(X_train, y_train) # Max depth vs Accuracy
    #
    # # Boosted decision tree experiment 1: Sample size vs Accuracy
    # boost_dt_experiment_1(X_train, y_train)
    #
    # # Boosted decision tree experiment 2: Leaf size vs Accuracy  (Pruning)
    boost_dt_experiment_2(X_train, y_train)
    # boost_dt_experiment_3(X_train, y_train)
    # boost_dt_experiment_4(X_train, y_train)
    #
    # # # ANN experiment 1: Sample size vs Accuracy
    # ann_experiment_1(X_train, y_train)
    #
    # # # ANN experiment 1: Sample size vs Accuracy
    # ann_experiment_2(X_train, y_train)
    # ann_experiment_3(X_train, y_train)
    # # #
    # # # # KNN experiment 1: Sample size vs Accuracy
    # knn_experiment_1(X_train, y_train)
    #
    # # # KNN experiment 1: Sample size vs Accuracy
    knn_experiment_2(X_train, y_train)
    # #
    # # # SVM experiment 1: Sample size vs Accuracy
    # svm_experiment_1(X_train, y_train)
    # #
    # # # SVM experiment 2: Sample size vs Accuracy
    # svm_experiment_2(X_train, y_train)
    # svm_experiment_3(X_train, y_train)











