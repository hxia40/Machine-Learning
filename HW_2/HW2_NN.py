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

def GD_valid_curve_learning_rate(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GD_learning_rate.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.logspace(-4, 1, 50):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=i,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(i, y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GD_learning_rate.txt', 'a')
            writer.write(str(i)+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def GD_valid_curve_max_atte(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GD_learning_rate.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=0.00032,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=int(i),
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GD_max_atte.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))
def GD_valid_curve_max_iter(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GD_learning_rate.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=int(i),
                                       bias=True, is_classifier=True, learning_rate=0.00032,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GD_max_iter.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()

def RHC_valid_curve_learning_rate(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/RHC_learning_rate.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1, 101, 2):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='random_hill_climb', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=i,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(i, y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/RHC_learning_rate.txt', 'a')
            writer.write(str(i)+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def RHC_valid_curve_max_atte(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/RHC_max_atte.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=0.01,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=int(i),
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/RHC_max_atte.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def RHC_valid_curve_max_iter(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/RHC_max_iter.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=int(i),
                                       bias=True, is_classifier=True, learning_rate=0.00032,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/RHC_max_iter.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()

def SA_valid_curve_learning_rate(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/SA_learning_rate.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.logspace(-2, 2, 50):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=i,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(i, y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/SA_learning_rate.txt', 'a')
            writer.write(str(i)+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def SA_valid_curve_max_atte(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/SA_max_atte.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=0.01,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=int(i),
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/SA_max_atte.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def SA_valid_curve_max_iter(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/SA_max_iter.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=int(i),
                                       bias=True, is_classifier=True, learning_rate=0.00032,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/SA_max_iter.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()

def GA_valid_curve_learning_rate(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GA_learning_rate.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.logspace(-2, 2, 50):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=i,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(i, y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GA_learning_rate.txt', 'a')
            writer.write(str(i)+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def GA_valid_curve_mutation_prob(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GA_mutation_prob.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(0.01, 1, 0.03):
            start_time = time.time()
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='genetic_alg', max_iters=10,
                                       bias=True, is_classifier=True, learning_rate=5,
                                       early_stopping=True,
                                       clip_max=10,
                                       max_attempts=35,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       pop_size = 200,
                                       mutation_prob = i,
                                       )
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print("time:", time.time()-start_time)
            print(i, y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GA_mutation_prob.txt', 'a')
            writer.write(str(i)+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))
def GA_valid_curve_max_atte(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GA_max_atte.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=100,
                                       bias=True, is_classifier=True, learning_rate=0.01,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=int(i),
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GA_max_atte.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()
def GA_valid_curve_max_iter(X, y, hidden_layer_sizes=(50, )):
    writer = open('ANN_curve/GA_max_iter.txt', 'w')
    rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    for train_index, validation_index in rs.split(X):

        X_train = X[train_index]
        X_valid = X[validation_index]
        y_train = y[train_index]
        y_valid = y[validation_index]
        # print "y_train", y_train, '\n', "valid:", y_test

        for i in np.arange(1,1000,20):
            # print("===========i===========", i)
            clf = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='gradient_descent', max_iters=int(i),
                                       bias=True, is_classifier=True, learning_rate=0.00032,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=100,
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       random_state=1)
            clf.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = clf.predict(X_train)
            # print("time:", time.time()-start)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # Predict labels for test set and assess accuracy
            y_valid_pred = clf.predict(X_valid)
            y_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            print(int(i), y_train_accuracy, y_valid_accuracy)

            writer = open('ANN_curve/GA_max_iter.txt', 'a')
            writer.write(str(int(i))+ str(",") + str(y_train_accuracy)+ str(",") + str(y_valid_accuracy)+str("\n"))
        print("\n")
        writer.write(str("\n\n"))




    # # recording
    # name = 'ann_learning_curve_epoch_post'
    # # n_alter_mean = np.mean(n_alter, axis=1)
    # train_scores_mean = np.mean(scores_train, axis=0)
    # train_scores_std = np.std(scores_train, axis=0)
    # test_scores_mean = np.mean(scores_test, axis=0)
    # test_scores_std = np.std(scores_test, axis=0)
    # DT_1 = open('{}/{}.txt'.format(dataset_name, name), 'w')
    # DT_1.write('{}/{}'.format(dataset_name, name))
    # DT_1.write("\n\n")
    # DT_1.write(str(n_alter))
    # DT_1.write("\n\n")
    # DT_1.write(str(train_scores_mean))
    # DT_1.write("\n\n")
    # DT_1.write(str(test_scores_mean))
    #
    # # plotting
    # plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(n_alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(n_alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(n_alter, train_scores_mean, color="r",
    #          label="Training score")
    # plt.plot(n_alter, test_scores_mean, color="g",
    #          label="Cross-validation score")
    # x_title = "Epochs"
    # y_title = "Score"
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    # plt.legend(loc="best")
    # plt.savefig('{}/{}.png'.format(dataset_name, name))
    # # plt.savefig('22222.png')
    # plt.gcf().clear()


def ann_learning_curve_size_GD_SH(X_train, y_train, X_test, y_test, hidden_nodes = [50]): # ANN experiment 1: Sample size vs Accuracy
    best_fitness = 0
    best_fitness_param = {
        "learning_rate": 0,
        "max_iters": 0}
    for learning_rate in np.arange(0.01, 10.01, 1):
        for max_iter in range(1, 501, 100):
            print("learning rate at:", learning_rate, "\n", "max_iters at:", max_iter)
            clf = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                                       algorithm='random_hill_climb', max_iters=max_iter,
                                       bias=True, is_classifier=True, learning_rate=learning_rate,
                                       early_stopping=True, clip_max=5, max_attempts=100,
                                       restarts=20
                                       )
            start_time = time.time()
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            y_fitness = accuracy_score(y_test, y_test_pred)
            if y_fitness > best_fitness:
                best_fitness = y_fitness
                best_fitness_param['learning_rate'] = learning_rate
                best_fitness_param['max_iters'] = max_iter
            print("time consumed:", time.time()-start_time)
    print("GD learning rate:", learning_rate)
    print("GD max iteration:", max_iter)
def GD_vld_curve_1(dataset_name, X_train, y_train, hidden_layer_sizes=(50, ), max_iter=100, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    # clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    clf = mlrose.NeuralNetwork(hidden_nodes=hidden_layer_sizes, activation='relu',
                               algorithm='gradient_descent', max_iters=max_iter,
                               bias=True, is_classifier=True, learning_rate=alpha,
                               early_stopping=True, clip_max=5, max_attempts=100,
                               restarts=20,
                               random_state=1)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = 5
    param_range = np.linspace(0.00001, 50, 25)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                                 param_name="learning_rate",
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring="accuracy",
                                                 n_jobs=1)
    end_time = time.time()
    difference = end_time - start_time
    print("ann_vld_curve_2", difference)

    recording_and_plotting(dataset_name, name="GD_vld_curve_1",
                           alter=param_range,
                           train=train_scores,
                           validation=test_scores, x_title="learning_rate", y_title="Score")
def ann_GD_vld_curve(dataset_name, X_train, y_train, hidden_nodes = [50], max_iter=500, alpha=0.0001): # ANN experiment 1: Sample size vs Accuracy
    clf = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                               algorithm='gradient_descent', max_iters=max_iter,
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
    '''Load and standardize data set ESR'''
    set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, 1:]
    set2 = set2.astype(int)

    # separating set2 into X and y, then train and test
    X2 = set2[:, :-1]
    scaler = preprocessing.StandardScaler()
    X2 = scaler.fit_transform(X2)
    y2 = set2[:, -1]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0, stratify=y2)
    set2_name = "ESR"

    # standardize the original data - this is important but usually neglected by newbies.
    scaler = preprocessing.StandardScaler()

    X_train_scaled = scaler.fit_transform(X2_train)
    X_test_scaled = scaler.transform(X2_test)

    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y2_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y2_test.reshape(-1, 1)).todense()
    set2_name = "ESR"

    GD_valid_curve_learning_rate(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    GD_valid_curve_max_atte(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    GD_valid_curve_max_iter(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))

    RHC_valid_curve_learning_rate(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    RHC_valid_curve_max_atte(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    RHC_valid_curve_max_iter(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))

    SA_valid_curve_learning_rate(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    SA_valid_curve_max_atte(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    SA_valid_curve_max_iter(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))

    GA_valid_curve_learning_rate(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    GA_valid_curve_mutation_prob(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    GA_valid_curve_max_atte(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))
    GA_valid_curve_max_iter(X_train_scaled, y_train_hot, hidden_layer_sizes=(50,))

    print("===========GD===========")
    time_list = []
    accuracy_list = []
    for i in range(10):
        start_time = time.time()
        clf_gd = mlrose.NeuralNetwork(hidden_nodes=[50,], activation='relu',
                                    algorithm='gradient_descent', max_iters=200,
                                    bias=True, is_classifier=True, learning_rate=0.00037,
                                    early_stopping=True,
                                    # clip_max=5,
                                    max_attempts=20,
                                    # restarts=20,
                                    # schedule= mlrose.GeomDecay(),
                                    # pop_size = 200,
                                    # mutation_prob = 0.1,
                                    )
        clf_gd.fit(X_test_scaled, y_test_hot)

        # Predict labels for test set and assess accuracy
        y_test_pred = clf_gd.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        accuracy_list.append(y_test_accuracy)
        time_list.append(time.time()-start_time)
    time_mean = np.mean(time_list)
    accuracy_mean = np.mean(accuracy_list)
    print("GD", time_mean, accuracy_mean)

    print("===========RHC===========")
    time_list = []
    accuracy_list = []
    for i in range(10):
        start_time = time.time()
        clf_rhc = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                       algorithm='random_hill_climb', max_iters=500,
                                       bias=True, is_classifier=True, learning_rate=10,
                                       early_stopping=True,
                                       # clip_max=5,
                                       max_attempts=500
                                       # restarts=20,
                                       # schedule= mlrose.GeomDecay(),
                                       # pop_size = 200,
                                       # mutation_prob = 0.1,
                                       )
        clf_rhc.fit(X_test_scaled, y_test_hot)

        # Predict labels for test set and assess accuracy
        y_test_pred = clf_rhc.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        accuracy_list.append(y_test_accuracy)
        time_list.append(time.time() - start_time)
    time_mean = np.mean(time_list)
    accuracy_mean = np.mean(accuracy_list)
    print("RHC", time_mean, accuracy_mean)

    print("===========SA===========")
    time_list = []
    accuracy_list = []
    for i in range(10):
        start_time = time.time()
        clf_sa = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                      algorithm='simulated_annealing', max_iters=500,
                                      bias=True, is_classifier=True, learning_rate=10,
                                      early_stopping=True,
                                      # clip_max=5,
                                      max_attempts=500,
                                      # restarts=20,
                                      schedule=mlrose.GeomDecay()
                                      # pop_size = 200,
                                      # mutation_prob = 0.1,
                                      )
        clf_sa.fit(X_test_scaled, y_test_hot)

        # Predict labels for test set and assess accuracy
        y_test_pred = clf_sa.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        accuracy_list.append(y_test_accuracy)
        time_list.append(time.time() - start_time)
    time_mean = np.mean(time_list)
    accuracy_mean = np.mean(accuracy_list)
    print("SA", time_mean, accuracy_mean)

    print("===========GA===========")
    time_list = []
    accuracy_list = []
    for i in range(10):
        start_time = time.time()
        clf_ga = mlrose.NeuralNetwork(hidden_nodes=[50, ], activation='relu',
                                      algorithm='genetic_alg', max_iters=200,
                                      bias=True, is_classifier=True, learning_rate=10,
                                      early_stopping=True,
                                      # clip_max=5,
                                      max_attempts=50,
                                      # restarts=20,
                                      # schedule= mlrose.GeomDecay()
                                      pop_size=200,
                                      mutation_prob=0.26
                                      )
        clf_ga.fit(X_test_scaled, y_test_hot)

        # Predict labels for test set and assess accuracy
        y_test_pred = clf_ga.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        accuracy_list.append(y_test_accuracy)
        time_list.append(time.time() - start_time)
    time_mean = np.mean(time_list)
    accuracy_mean = np.mean(accuracy_list)
    print("GA", time_mean, accuracy_mean)




