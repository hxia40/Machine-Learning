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
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit, cross_val_score, \
    cross_validate
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

if __name__ == "__main__":
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

    ''' clutering '''
    ###### old data + clustered data ######

    km1 = kmeans(n_clusters=3)
    data1_X_km = np.concatenate((data1_X_train, km1.fit_transform(data1_X_train)), axis=1)
    data1_X_km_test = np.concatenate((data1_X_test, km1.transform(data1_X_test)), axis=1)

    print km1.transform(data1_X_test).shape
    km2 = kmeans(n_clusters=3)
    data2_X_km = np.concatenate((data2_X_train, km2.fit_transform(data2_X_train)), axis=1)
    data2_X_km_test = np.concatenate((data2_X_test, km2.transform(data2_X_test)), axis=1)

    gmm1 = GMM(n_components=3)
    gmm1.fit(data1_X_train)
    print data1_X_train.shape
    print gmm1.predict(data1_X_train).shape

    data1_X_gmm = np.concatenate((data1_X_train, gmm1.predict(data1_X_train)), axis=1)
    data1_X_gmm_test = np.concatenate((data1_X_test, gmm1.predict(data1_X_test)), axis=1)

    gmm2 = GMM(n_components=3)
    gmm2.fit(data2_X_train)
    data2_X_gmm = np.concatenate((data2_X_train, gmm2.transform(data2_X_train)), axis=1)
    data2_X_gmm_test = np.concatenate((data2_X_test, gmm2.transform(data2_X_test)), axis=1)





    ''' neuron network '''
    file_5 = open('part_5.txt', 'w')
    # ann_learning_curve_size_post(set1_name, data1_X_train, data1_y_train, hidden_layer_sizes=(50,), alpha=6.25)
    # ann_learning_curve_size_post(set2_name, data2_X_train, data2_y_train, hidden_layer_sizes=(50, ), alpha=0.417)

    ########## MNIST###########

    ann1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=6.25, random_state=1)

    score_orig_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_train, data1_y_train).predict(data1_X_test))
    score_km_1 = accuracy_score(data1_y_test, ann1.fit(data1_X_km, data1_y_train).predict(data1_X_km_test))


    file_5.write('score_orig_1:')
    file_5.write(str(score_orig_1))
    file_5.write('\n')
    file_5.write('score_km_1:')
    file_5.write(str(score_km_1))
    file_5.write('\n')


    ########## ESR ###########


print "========== END =========="
