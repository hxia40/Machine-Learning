import numpy as np
import pandas as pd
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy as sp
import time
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit, cross_val_score, cross_validate
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def recording_and_plotting(dataset_name, name, alter, train, validation,
                           x_title="Sample size", y_title="Score"):
    train_scores_mean = np.mean(train, axis=1)
    train_scores_std = np.std(train, axis=1)
    test_scores_mean = np.mean(validation, axis=1)
    test_scores_std = np.std(validation, axis=1)

    # recording
    DT_1 = open('part4_{}_{}.txt'.format(dataset_name, name), 'w')
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
    plt.savefig('part4_{}_{}.png'.format(dataset_name, name))
    plt.gcf().clear()

def ann_learning_curve_size_post(dataset_name, X_train, y_train, hidden_layer_sizes=(5, ), max_iter=500, alpha=0.0001):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha, random_state=1)
    start_time = time.time()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # cv = None

    train_sizes = np.linspace(.01, 1.0, 2)
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

def score_time(dataset_name, clf_name, clf, X_train, X_test, y_train, y_test):
    start_time = time.time()
    score = 0
    difference = 0
    for i in range(1):
        clf.fit(X_train, y_train).predict(X_test)
        score += accuracy_score(y_test, clf.fit(X_train, y_train).predict(X_test))
        end_time = time.time()
        difference += (end_time - start_time)
    txt = open('part_4', 'a')
    txt.write('{}_{} score:'.format(dataset_name, clf_name))
    txt.write(str(score/1))
    txt.write("\n")
    txt.write('{}_{} time:'.format(dataset_name, clf_name))
    txt.write(str(difference/1))
    txt.write("\n\n")

if __name__=="__main__":
    
    # Data Loading & Preprocessing
    scaler = preprocessing.StandardScaler()
    one_hot = preprocessing.OneHotEncoder()

    '''Load and standardize data set MNIST'''
    set1_name = "mnist"

    train = np.genfromtxt('fashion-mnist_train_minor.csv', delimiter=',')[1:, :]
    test = np.genfromtxt('fashion-mnist_test_minor.csv', delimiter=',')[1:, :]

    data1_X_train = train[:, 1:]
    data1_y_train = train[:, 0]
    data1_X_test = test[:, 1:]
    data1_y_test = test[:, 0]

    data1_X_train = scaler.fit_transform(data1_X_train)
    data1_X_test = scaler.transform(data1_X_test)

    '''Load and standardize data set ESR'''
    set2_name = "ESR"
    set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, 1:]

    set2 = set2.astype(int)

    data2_X = set2[:, :-1]
    data2_X = scaler.fit_transform(data2_X)
    data2_y = set2[:, -1]
    data2_X = scaler.fit_transform(data2_X)
    data2_X_train, data2_X_test, data2_y_train, data2_y_test = train_test_split(data2_X, data2_y,
                                                                                test_size=0.2,
                                                                                random_state=0,
                                                                                stratify=data2_y)

    ''' dimension reduction '''
    ###### old data + feature-reduced data ######

    pca1 = PCA(n_components = 20)
    data1_X_pca = np.concatenate((data1_X_train, pca1.fit_transform(data1_X_train)), axis = 1)
    data1_X_pca_test = np.concatenate((data1_X_test, pca1.transform(data1_X_test)), axis = 1)
    pca2 = PCA(n_components = 90)
    data2_X_pca = np.concatenate((data2_X_train, pca2.fit_transform(data2_X_train)), axis = 1)
    data2_X_pca_test = np.concatenate((data2_X_test, pca2.transform(data2_X_test)), axis = 1)

    ica1 = FastICA(n_components = 20)
    data1_X_ica = np.concatenate((data1_X_train, ica1.fit_transform(data1_X_train)), axis = 1)
    data1_X_ica_test = np.concatenate((data1_X_test, ica1.transform(data1_X_test)), axis = 1)
    ica2 = FastICA(n_components = 90)
    data2_X_ica = np.concatenate((data2_X_train, ica2.fit_transform(data2_X_train)), axis = 1)
    data2_X_ica_test = np.concatenate((data2_X_test, ica2.transform(data2_X_test)), axis = 1)

    grp1 = GaussianRandomProjection(n_components = 20)
    data1_X_grp = np.concatenate((data1_X_train, grp1.fit_transform(data1_X_train)), axis=1)
    data1_X_grp_test = np.concatenate((data1_X_test, grp1.transform(data1_X_test)), axis=1)
    grp2 = GaussianRandomProjection(n_components = 90)
    data2_X_grp = np.concatenate((data2_X_train, grp2.fit_transform(data2_X_train)), axis=1)
    data2_X_grp_test = np.concatenate((data2_X_test, grp2.transform(data2_X_test)), axis=1)

    fa1 = FeatureAgglomeration(n_clusters = 20)
    data1_X_fa = np.concatenate((data1_X_train, fa1.fit_transform(data1_X_train)), axis=1)
    data1_X_fa_test = np.concatenate((data1_X_test, fa1.transform(data1_X_test)), axis=1)
    fa2 = FeatureAgglomeration(n_clusters = 90)
    data2_X_fa = np.concatenate((data2_X_train, fa2.fit_transform(data2_X_train)), axis=1)
    data2_X_fa_test = np.concatenate((data2_X_test, fa2.transform(data2_X_test)), axis=1)

    ###### feature-reduced data only ######

    # pca1 = PCA(n_components = 20)
    # data1_X_pca = pca1.fit_transform(data1_X_train)
    # data1_X_pca_test = pca1.transform(data1_X_test)
    # pca2 = GaussianRandomProjection(n_components = 90)
    # data2_X_pca = pca2.fit_transform(data2_X_train)
    # data2_X_pca_test = pca2.transform(data2_X_test)

    # ica1 = FastICA(n_components = 20)
    # data1_X_ica = ica1.fit_transform(data1_X_train)
    # data1_X_ica_test = ica1.transform(data1_X_test)
    # ica2 = GaussianRandomProjection(n_components = 90)
    # data2_X_ica = ica2.fit_transform(data2_X_train)
    # data2_X_ica_test = ica2.transform(data2_X_test)

    # grp1 = GaussianRandomProjection(n_components = 20)
    # data1_X_grp = grp1.fit_transform(data1_X_train)
    # data1_X_grp_test = grp1.transform(data1_X_test)
    # grp2 = GaussianRandomProjection(n_components = 90)
    # data2_X_grp = grp2.fit_transform(data2_X_train)
    # data2_X_grp_test = grp2.transform(data2_X_test)
    #
    # fa1 = FeatureAgglomeration(n_clusters = 20)
    # data1_X_fa = fa1.fit_transform(data1_X_train)
    # data1_X_fa_test = fa1.transform(data1_X_test)
    # fa2 = FeatureAgglomeration(n_clusters = 90)
    # data2_X_fa = fa2.fit_transform(data2_X_train)
    # data2_X_fa_test = fa2.transform(data2_X_test)

    ''' neuron network '''
    file_4 = open('part_4.txt', 'w')
    # ann_learning_curve_size_post(set1_name, data1_X_train, data1_y_train, hidden_layer_sizes=(50,), alpha=6.25)
    # ann_learning_curve_size_post(set2_name, data2_X_train, data2_y_train, hidden_layer_sizes=(50, ), alpha=0.417)

    ########## MNIST###########

    ann1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=6.25, random_state=1)

    score_orig_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_train, data1_y_train).predict(data1_X_test))
    score_pca_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_pca, data1_y_train).predict(data1_X_pca_test))
    score_ica_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_ica, data1_y_train).predict(data1_X_ica_test))
    score_grp_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_grp, data1_y_train).predict(data1_X_grp_test))
    score_fa_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_fa, data1_y_train).predict(data1_X_fa_test))

    file_4.write('score_orig_1:')
    file_4.write(str(score_orig_1))
    file_4.write('\n')
    file_4.write('score_pca_1:')
    file_4.write(str(score_pca_1))
    file_4.write('\n')
    file_4.write('score_ica_1:')
    file_4.write(str(score_ica_1))
    file_4.write('\n')
    file_4.write('score_grp_1:')
    file_4.write(str(score_grp_1))
    file_4.write('\n')
    file_4.write('score_fa_1:')
    file_4.write(str(score_fa_1))
    file_4.write('\n')

    ########## ESR ###########

    ann2 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=0.417, random_state=1)

    score_orig_2 = accuracy_score(data2_y_test, ann2.fit(data2_X_train, data2_y_train).predict(data2_X_test))
    score_pca_2 = accuracy_score(data2_y_test, ann2.fit(data2_X_pca, data2_y_train).predict(data2_X_pca_test))
    score_ica_2 = accuracy_score(data2_y_test, ann2.fit(data2_X_ica, data2_y_train).predict(data2_X_ica_test))
    score_grp_2 = accuracy_score(data2_y_test, ann2.fit(data2_X_grp, data2_y_train).predict(data2_X_grp_test))
    score_fa_2 = accuracy_score(data2_y_test, ann2.fit(data2_X_fa, data2_y_train).predict(data2_X_fa_test))

    file_4.write('score_orig_2:')
    file_4.write(str(score_orig_2))
    file_4.write('\n')
    file_4.write('score_pca_2:')
    file_4.write(str(score_pca_2))
    file_4.write('\n')
    file_4.write('score_ica_2:')
    file_4.write(str(score_ica_2))
    file_4.write('\n')
    file_4.write('score_grp_2:')
    file_4.write(str(score_grp_2))
    file_4.write('\n')
    file_4.write('score_fa_2:')
    file_4.write(str(score_fa_2))
    file_4.write('\n')

    # score_time("MNIST", "ANN",
    #            MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500, alpha=6.25, random_state=1),
    #            data1_X_train, data1_X_test, data1_y_train, data1_y_test)
    # score_time("ESR", "ANN",
    #            MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500, alpha=0.417, random_state=1),
    #            data1_X_train, data1_X_test, data1_y_train, data1_y_test)


print "========== END =========="
