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

    file_2 = open('file_2.txt', 'w')

    ''' dimension reduction '''

    pca1 = PCA(n_components = 20)
    data1_X_pca = pca1.fit_transform(data1_X_train)
    data1_X_pca_test = pca1.transform(data1_X_test)
    pca2 = PCA(n_components = 90)
    data2_X_pca = pca2.fit_transform(data2_X_train)
    data2_X_pca_test = pca2.transform(data2_X_test)

    ica1 = FastICA(n_components = 20)
    data1_X_ica = ica1.fit_transform(data1_X_train)
    data1_X_ica_test = ica1.transform(data1_X_test)
    ica2 = FastICA(n_components = 90)
    data2_X_ica = ica2.fit_transform(data2_X_train)
    data2_X_ica_test = ica2.transform(data2_X_test)

    grp1 = GaussianRandomProjection(n_components = 20)
    data1_X_grp = grp1.fit_transform(data1_X_train)
    data1_X_grp_test = grp1.transform(data1_X_test)
    grp2 = GaussianRandomProjection(n_components = 90)
    data2_X_grp = grp2.fit_transform(data2_X_train)
    data2_X_grp_test = grp2.transform(data2_X_test)

    fa1 = FeatureAgglomeration(n_clusters = 20)
    data1_X_fa = fa1.fit_transform(data1_X_train)
    data1_X_fa_test = fa1.transform(data1_X_test)
    fa2 = FeatureAgglomeration(n_clusters = 90)
    data2_X_fa = fa2.fit_transform(data2_X_train)
    data2_X_fa_test = fa2.transform(data2_X_test)

    ''' clustering '''

    clusters = np.logspace(0.5, 2, num=10, endpoint=True, base=10.0, dtype=None)
    for i in range(0, len(clusters)):
        clusters[i] = int(clusters[i])
    print clusters

    temp = len(clusters)
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    sil_kmean_pca_train_1 = np.zeros(temp)
    sil_gmm_pca_train_1 = np.zeros(temp)
    sil_kmean_pca_test_1 = np.zeros(temp)
    sil_gmm_pca_test_1 = np.zeros(temp)
    sil_kmean_ica_train_1 = np.zeros(temp)
    sil_gmm_ica_train_1 = np.zeros(temp)
    sil_kmean_ica_test_1 = np.zeros(temp)
    sil_gmm_ica_test_1 = np.zeros(temp)
    sil_kmean_grp_train_1 = np.zeros(temp)
    sil_gmm_grp_train_1 = np.zeros(temp)
    sil_kmean_grp_test_1 = np.zeros(temp)
    sil_gmm_grp_test_1 = np.zeros(temp)
    sil_kmean_fa_train_1 = np.zeros(temp)
    sil_gmm_fa_train_1 = np.zeros(temp)
    sil_kmean_fa_test_1 = np.zeros(temp)
    sil_gmm_fa_test_1 = np.zeros(temp)

    sil_kmean_pca_train_2 = np.zeros(temp)
    sil_gmm_pca_train_2 = np.zeros(temp)
    sil_kmean_pca_test_2 = np.zeros(temp)
    sil_gmm_pca_test_2 = np.zeros(temp)
    sil_kmean_ica_train_2 = np.zeros(temp)
    sil_gmm_ica_train_2 = np.zeros(temp)
    sil_kmean_ica_test_2 = np.zeros(temp)
    sil_gmm_ica_test_2 = np.zeros(temp)
    sil_kmean_grp_train_2 = np.zeros(temp)
    sil_gmm_grp_train_2 = np.zeros(temp)
    sil_kmean_grp_test_2 = np.zeros(temp)
    sil_gmm_grp_test_2 = np.zeros(temp)
    sil_kmean_fa_train_2 = np.zeros(temp)
    sil_gmm_fa_train_2 = np.zeros(temp)
    sil_kmean_fa_test_2 = np.zeros(temp)
    sil_gmm_fa_test_2 = np.zeros(temp)

    db_kmean_pca_train_1 = np.zeros(temp)
    db_gmm_pca_train_1 = np.zeros(temp)
    db_kmean_pca_test_1 = np.zeros(temp)
    db_gmm_pca_test_1 = np.zeros(temp)
    db_kmean_ica_train_1 = np.zeros(temp)
    db_gmm_ica_train_1 = np.zeros(temp)
    db_kmean_ica_test_1 = np.zeros(temp)
    db_gmm_ica_test_1 = np.zeros(temp)
    db_kmean_grp_train_1 = np.zeros(temp)
    db_gmm_grp_train_1 = np.zeros(temp)
    db_kmean_grp_test_1 = np.zeros(temp)
    db_gmm_grp_test_1 = np.zeros(temp)
    db_kmean_fa_train_1 = np.zeros(temp)
    db_gmm_fa_train_1 = np.zeros(temp)
    db_kmean_fa_test_1 = np.zeros(temp)
    db_gmm_fa_test_1 = np.zeros(temp)

    db_kmean_pca_train_2 = np.zeros(temp)
    db_gmm_pca_train_2 = np.zeros(temp)
    db_kmean_pca_test_2 = np.zeros(temp)
    db_gmm_pca_test_2 = np.zeros(temp)
    db_kmean_ica_train_2 = np.zeros(temp)
    db_gmm_ica_train_2 = np.zeros(temp)
    db_kmean_ica_test_2 = np.zeros(temp)
    db_gmm_ica_test_2 = np.zeros(temp)
    db_kmean_grp_train_2 = np.zeros(temp)
    db_gmm_grp_train_2 = np.zeros(temp)
    db_kmean_grp_test_2 = np.zeros(temp)
    db_gmm_grp_test_2 = np.zeros(temp)
    db_kmean_fa_train_2 = np.zeros(temp)
    db_gmm_fa_train_2 = np.zeros(temp)
    db_kmean_fa_test_2 = np.zeros(temp)
    db_gmm_fa_test_2 = np.zeros(temp)

    for i in range(0, temp):
        km.set_params(n_clusters = int(clusters[i]))
        gmm.set_params(n_components = int(clusters[i]))

        ########## MNIST-pca ###########

        km.fit(data1_X_pca)
        gmm.fit(data1_X_pca)

        km_pca_train_labels_1 = km.predict(data1_X_pca)
        gmm_pca_train_labels_1 = gmm.predict(data1_X_pca)
        km_pca_test_labels_1 = km.predict(data1_X_pca_test)
        gmm_pca_test_labels_1 = gmm.predict(data1_X_pca_test)

        sil_kmean_pca_train_1[i] = metrics.silhouette_score(data1_X_pca, km_pca_train_labels_1)
        sil_gmm_pca_train_1[i] = metrics.silhouette_score(data1_X_pca, gmm_pca_train_labels_1)
        sil_kmean_pca_test_1[i] = metrics.silhouette_score(data1_X_pca_test, km_pca_test_labels_1)
        sil_gmm_pca_test_1[i] = metrics.silhouette_score(data1_X_pca_test, gmm_pca_test_labels_1)

        db_kmean_pca_train_1[i] = metrics.davies_bouldin_score(data1_X_pca, km_pca_train_labels_1)
        db_gmm_pca_train_1[i] = metrics.davies_bouldin_score(data1_X_pca, gmm_pca_train_labels_1)
        db_kmean_pca_test_1[i] = metrics.davies_bouldin_score(data1_X_pca_test, km_pca_test_labels_1)
        db_gmm_pca_test_1[i] = metrics.davies_bouldin_score(data1_X_pca_test, gmm_pca_test_labels_1)

        ########## ESR-pca ###########

        km.fit(data2_X_pca)
        gmm.fit(data2_X_pca)

        km_pca_train_labels_2 = km.predict(data2_X_pca)
        gmm_pca_train_labels_2 = gmm.predict(data2_X_pca)
        km_pca_test_labels_2 = km.predict(data2_X_pca_test)
        gmm_pca_test_labels_2 = gmm.predict(data2_X_pca_test)

        sil_kmean_pca_train_2[i] = metrics.silhouette_score(data2_X_pca, km_pca_train_labels_2)
        sil_gmm_pca_train_2[i] = metrics.silhouette_score(data2_X_pca, gmm_pca_train_labels_2)
        sil_kmean_pca_test_2[i] = metrics.silhouette_score(data2_X_pca_test, km_pca_test_labels_2)
        sil_gmm_pca_test_2[i] = metrics.silhouette_score(data2_X_pca_test, gmm_pca_test_labels_2)

        db_kmean_pca_train_2[i] = metrics.davies_bouldin_score(data2_X_pca, km_pca_train_labels_2)
        db_gmm_pca_train_2[i] = metrics.davies_bouldin_score(data2_X_pca, gmm_pca_train_labels_2)
        db_kmean_pca_test_2[i] = metrics.davies_bouldin_score(data2_X_pca_test, km_pca_test_labels_2)
        db_gmm_pca_test_2[i] = metrics.davies_bouldin_score(data2_X_pca_test, gmm_pca_test_labels_2)

        ########## MNIST-ica ###########

        km.fit(data1_X_ica)
        gmm.fit(data1_X_ica)

        km_ica_train_labels_1 = km.predict(data1_X_ica)
        gmm_ica_train_labels_1 = gmm.predict(data1_X_ica)
        km_ica_test_labels_1 = km.predict(data1_X_ica_test)
        gmm_ica_test_labels_1 = gmm.predict(data1_X_ica_test)

        sil_kmean_ica_train_1[i] = metrics.silhouette_score(data1_X_ica, km_ica_train_labels_1)
        sil_gmm_ica_train_1[i] = metrics.silhouette_score(data1_X_ica, gmm_ica_train_labels_1)
        sil_kmean_ica_test_1[i] = metrics.silhouette_score(data1_X_ica_test, km_ica_test_labels_1)
        sil_gmm_ica_test_1[i] = metrics.silhouette_score(data1_X_ica_test, gmm_ica_test_labels_1)

        db_kmean_ica_train_1[i] = metrics.davies_bouldin_score(data1_X_ica, km_ica_train_labels_1)
        db_gmm_ica_train_1[i] = metrics.davies_bouldin_score(data1_X_ica, gmm_ica_train_labels_1)
        db_kmean_ica_test_1[i] = metrics.davies_bouldin_score(data1_X_ica_test, km_ica_test_labels_1)
        db_gmm_ica_test_1[i] = metrics.davies_bouldin_score(data1_X_ica_test, gmm_ica_test_labels_1)

        ########## ESR-ica ###########

        km.fit(data2_X_ica)
        gmm.fit(data2_X_ica)

        km_ica_train_labels_2 = km.predict(data2_X_ica)
        gmm_ica_train_labels_2 = gmm.predict(data2_X_ica)
        km_ica_test_labels_2 = km.predict(data2_X_ica_test)
        gmm_ica_test_labels_2 = gmm.predict(data2_X_ica_test)

        sil_kmean_ica_train_2[i] = metrics.silhouette_score(data2_X_ica, km_ica_train_labels_2)
        sil_gmm_ica_train_2[i] = metrics.silhouette_score(data2_X_ica, gmm_ica_train_labels_2)
        sil_kmean_ica_test_2[i] = metrics.silhouette_score(data2_X_ica_test, km_ica_test_labels_2)
        sil_gmm_ica_test_2[i] = metrics.silhouette_score(data2_X_ica_test, gmm_ica_test_labels_2)

        db_kmean_ica_train_2[i] = metrics.davies_bouldin_score(data2_X_ica, km_ica_train_labels_2)
        db_gmm_ica_train_2[i] = metrics.davies_bouldin_score(data2_X_ica, gmm_ica_train_labels_2)
        db_kmean_ica_test_2[i] = metrics.davies_bouldin_score(data2_X_ica_test, km_ica_test_labels_2)
        db_gmm_ica_test_2[i] = metrics.davies_bouldin_score(data2_X_ica_test, gmm_ica_test_labels_2)

        ########## MNIST-grp ###########

        km.fit(data1_X_grp)
        gmm.fit(data1_X_grp)

        km_grp_train_labels_1 = km.predict(data1_X_grp)
        gmm_grp_train_labels_1 = gmm.predict(data1_X_grp)
        km_grp_test_labels_1 = km.predict(data1_X_grp_test)
        gmm_grp_test_labels_1 = gmm.predict(data1_X_grp_test)

        sil_kmean_grp_train_1[i] = metrics.silhouette_score(data1_X_grp, km_grp_train_labels_1)
        sil_gmm_grp_train_1[i] = metrics.silhouette_score(data1_X_grp, gmm_grp_train_labels_1)
        sil_kmean_grp_test_1[i] = metrics.silhouette_score(data1_X_grp_test, km_grp_test_labels_1)
        sil_gmm_grp_test_1[i] = metrics.silhouette_score(data1_X_grp_test, gmm_grp_test_labels_1)

        db_kmean_grp_train_1[i] = metrics.davies_bouldin_score(data1_X_grp, km_grp_train_labels_1)
        db_gmm_grp_train_1[i] = metrics.davies_bouldin_score(data1_X_grp, gmm_grp_train_labels_1)
        db_kmean_grp_test_1[i] = metrics.davies_bouldin_score(data1_X_grp_test, km_grp_test_labels_1)
        db_gmm_grp_test_1[i] = metrics.davies_bouldin_score(data1_X_grp_test, gmm_grp_test_labels_1)

        ########## ESR-grp ###########

        km.fit(data2_X_grp)
        gmm.fit(data2_X_grp)

        km_grp_train_labels_2 = km.predict(data2_X_grp)
        gmm_grp_train_labels_2 = gmm.predict(data2_X_grp)
        km_grp_test_labels_2 = km.predict(data2_X_grp_test)
        gmm_grp_test_labels_2 = gmm.predict(data2_X_grp_test)

        sil_kmean_grp_train_2[i] = metrics.silhouette_score(data2_X_grp, km_grp_train_labels_2)
        sil_gmm_grp_train_2[i] = metrics.silhouette_score(data2_X_grp, gmm_grp_train_labels_2)
        sil_kmean_grp_test_2[i] = metrics.silhouette_score(data2_X_grp_test, km_grp_test_labels_2)
        sil_gmm_grp_test_2[i] = metrics.silhouette_score(data2_X_grp_test, gmm_grp_test_labels_2)

        db_kmean_grp_train_2[i] = metrics.davies_bouldin_score(data2_X_grp, km_grp_train_labels_2)
        db_gmm_grp_train_2[i] = metrics.davies_bouldin_score(data2_X_grp, gmm_grp_train_labels_2)
        db_kmean_grp_test_2[i] = metrics.davies_bouldin_score(data2_X_grp_test, km_grp_test_labels_2)
        db_gmm_grp_test_2[i] = metrics.davies_bouldin_score(data2_X_grp_test, gmm_grp_test_labels_2)

        ########## MNIST-fa ###########

        km.fit(data1_X_fa)
        gmm.fit(data1_X_fa)

        km_fa_train_labels_1 = km.predict(data1_X_fa)
        gmm_fa_train_labels_1 = gmm.predict(data1_X_fa)
        km_fa_test_labels_1 = km.predict(data1_X_fa_test)
        gmm_fa_test_labels_1 = gmm.predict(data1_X_fa_test)

        sil_kmean_fa_train_1[i] = metrics.silhouette_score(data1_X_fa, km_fa_train_labels_1)
        sil_gmm_fa_train_1[i] = metrics.silhouette_score(data1_X_fa, gmm_fa_train_labels_1)
        sil_kmean_fa_test_1[i] = metrics.silhouette_score(data1_X_fa_test, km_fa_test_labels_1)
        sil_gmm_fa_test_1[i] = metrics.silhouette_score(data1_X_fa_test, gmm_fa_test_labels_1)

        db_kmean_fa_train_1[i] = metrics.davies_bouldin_score(data1_X_fa, km_fa_train_labels_1)
        db_gmm_fa_train_1[i] = metrics.davies_bouldin_score(data1_X_fa, gmm_fa_train_labels_1)
        db_kmean_fa_test_1[i] = metrics.davies_bouldin_score(data1_X_fa_test, km_fa_test_labels_1)
        db_gmm_fa_test_1[i] = metrics.davies_bouldin_score(data1_X_fa_test, gmm_fa_test_labels_1)

        ########## ESR-fa ###########

        km.fit(data2_X_fa)
        gmm.fit(data2_X_fa)

        km_fa_train_labels_2 = km.predict(data2_X_fa)
        gmm_fa_train_labels_2 = gmm.predict(data2_X_fa)
        km_fa_test_labels_2 = km.predict(data2_X_fa_test)
        gmm_fa_test_labels_2 = gmm.predict(data2_X_fa_test)

        sil_kmean_fa_train_2[i] = metrics.silhouette_score(data2_X_fa, km_fa_train_labels_2)
        sil_gmm_fa_train_2[i] = metrics.silhouette_score(data2_X_fa, gmm_fa_train_labels_2)
        sil_kmean_fa_test_2[i] = metrics.silhouette_score(data2_X_fa_test, km_fa_test_labels_2)
        sil_gmm_fa_test_2[i] = metrics.silhouette_score(data2_X_fa_test, gmm_fa_test_labels_2)

        db_kmean_fa_train_2[i] = metrics.davies_bouldin_score(data2_X_fa, km_fa_train_labels_2)
        db_gmm_fa_train_2[i] = metrics.davies_bouldin_score(data2_X_fa, gmm_fa_train_labels_2)
        db_kmean_fa_test_2[i] = metrics.davies_bouldin_score(data2_X_fa_test, km_fa_test_labels_2)
        db_gmm_fa_test_2[i] = metrics.davies_bouldin_score(data2_X_fa_test, gmm_fa_test_labels_2)

        '''==========='''

        print i+1

    file_3 = open('part_3.txt','w')

    ### MNIST - sil - pca ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil pca train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_pca_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean sil pca test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_pca_test_1[i])
    file_3.write("\n")

    file_3.write("gmm sil pca train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_pca_train_1[i])
    file_3.write("\n")

    file_3.write("gmm sil pca test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_pca_test_1[i])
    file_3.write("\n")

    ### ESR - sil - pca ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil pca train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_pca_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean sil pca test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_pca_test_2[i])
    file_3.write("\n")

    file_3.write("gmm sil pca train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_pca_train_2[i])
    file_3.write("\n")

    file_3.write("gmm sil pca test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_pca_test_2[i])
    file_3.write("\n")

    ### MNIST - db - pca ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db pca train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_pca_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean db pca test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_pca_test_1[i])
    file_3.write("\n")

    file_3.write("gmm db pca train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_pca_train_1[i])
    file_3.write("\n")

    file_3.write("gmm db pca test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_pca_test_1[i])
    file_3.write("\n")

    ### ESR - db - pca ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db pca train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_pca_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean db pca test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_pca_test_2[i])
    file_3.write("\n")

    file_3.write("gmm db pca train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_pca_train_2[i])
    file_3.write("\n")

    file_3.write("gmm db pca test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_pca_test_2[i])
    file_3.write("\n")

    ### MNIST - sil - ica ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil ica train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_ica_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean sil ica test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_ica_test_1[i])
    file_3.write("\n")

    file_3.write("gmm sil ica train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_ica_train_1[i])
    file_3.write("\n")

    file_3.write("gmm sil ica test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_ica_test_1[i])
    file_3.write("\n")

    ### ESR - sil - ica ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil ica train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_ica_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean sil ica test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_ica_test_2[i])
    file_3.write("\n")

    file_3.write("gmm sil ica train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_ica_train_2[i])
    file_3.write("\n")

    file_3.write("gmm sil ica test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_ica_test_2[i])
    file_3.write("\n")

    ### MNIST - db - ica ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db ica train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_ica_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean db ica test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_ica_test_1[i])
    file_3.write("\n")

    file_3.write("gmm db ica train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_ica_train_1[i])
    file_3.write("\n")

    file_3.write("gmm db ica test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_ica_test_1[i])
    file_3.write("\n")

    ### ESR - db - ica ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db ica train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_ica_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean db ica test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_ica_test_2[i])
    file_3.write("\n")

    file_3.write("gmm db ica train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_ica_train_2[i])
    file_3.write("\n")

    file_3.write("gmm db ica test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_ica_test_2[i])
    file_3.write("\n")

    ### MNIST - sil - grp ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil grp train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_grp_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean sil grp test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_grp_test_1[i])
    file_3.write("\n")

    file_3.write("gmm sil grp train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_grp_train_1[i])
    file_3.write("\n")

    file_3.write("gmm sil grp test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_grp_test_1[i])
    file_3.write("\n")

    ### ESR - sil - grp ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil grp train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_grp_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean sil grp test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_grp_test_2[i])
    file_3.write("\n")

    file_3.write("gmm sil grp train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_grp_train_2[i])
    file_3.write("\n")

    file_3.write("gmm sil grp test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_grp_test_2[i])
    file_3.write("\n")

    ### MNIST - db - grp ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db grp train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_grp_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean db grp test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_grp_test_1[i])
    file_3.write("\n")

    file_3.write("gmm db grp train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_grp_train_1[i])
    file_3.write("\n")

    file_3.write("gmm db grp test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_grp_test_1[i])
    file_3.write("\n")

    ### ESR - db - grp ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db grp train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_grp_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean db grp test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_grp_test_2[i])
    file_3.write("\n")

    file_3.write("gmm db grp train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_grp_train_2[i])
    file_3.write("\n")

    file_3.write("gmm db grp test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_grp_test_2[i])
    file_3.write("\n")

    ### MNIST - sil - fa ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil fa train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_fa_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean sil fa test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_kmean_fa_test_1[i])
    file_3.write("\n")

    file_3.write("gmm sil fa train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_fa_train_1[i])
    file_3.write("\n")

    file_3.write("gmm sil fa test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % sil_gmm_fa_test_1[i])
    file_3.write("\n")

    ### ESR - sil - fa ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean sil fa train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_fa_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean sil fa test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_kmean_fa_test_2[i])
    file_3.write("\n")

    file_3.write("gmm sil fa train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_fa_train_2[i])
    file_3.write("\n")

    file_3.write("gmm sil fa test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % sil_gmm_fa_test_2[i])
    file_3.write("\n")

    ### MNIST - db - fa ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db fa train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_fa_train_1[i])
    file_3.write("\n")

    file_3.write("k-mean db fa test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_kmean_fa_test_1[i])
    file_3.write("\n")

    file_3.write("gmm db fa train 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_fa_train_1[i])
    file_3.write("\n")

    file_3.write("gmm db fa test 1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % db_gmm_fa_test_1[i])
    file_3.write("\n")

    ### ESR - db - fa ###

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean db fa train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_fa_train_2[i])
    file_3.write("\n")

    file_3.write("k-mean db fa test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_kmean_fa_test_2[i])
    file_3.write("\n")

    file_3.write("gmm db fa train 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_fa_train_2[i])
    file_3.write("\n")

    file_3.write("gmm db fa test 2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%2.9f" % db_gmm_fa_test_2[i])
    file_3.write("\n")

    file_3.close()


print "========== END =========="
