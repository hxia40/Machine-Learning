import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import time

if __name__=="__main__":

    # Data Loading & Preprocessing

    data1 = np.loadtxt(open("default_of_credit_card_clients.csv", "rb"), delimiter=";", skiprows=1)
    data2 = np.loadtxt(open("FFdFactorsPs.csv", "rb"), delimiter=",", skiprows=1)

    data1_X = data1[: , 0:-1]
    data1_Y = data1[: , -1]

    start_date = np.min(np.argwhere(data2[: , 0] == 19700102))
    data2_X = np.column_stack((data2[ (start_date) : -1 , (2, 3)], data2[ (start_date - 1) : -2 , 1]))
    data2_Y = data2[ (start_date) : -1 , 1]
    data2_Y = (data2_Y >= 0) * 1
    
    n1 = data1_Y.shape[0]
    n2 = data2_Y.shape[0]
    train_n1 = int(round(n1 * 0.6, 0))
    train_n2 = int(round(n2 * 0.6, 0))
    test_n1 = n1 - train_n1
    test_n2 = n2 - train_n2

    data1_X_train = data1_X[0:train_n1, : ]
    data1_Y_train = data1_Y[0:train_n1]
    data1_X_test = data1_X[train_n1 : -1 , : ]
    data1_Y_test = data1_Y[train_n1 : -1 ]

    data2_X_train = data2_X[0:train_n2, : ]
    data2_Y_train = data2_Y[0:train_n2]
    data2_X_test = data2_X[train_n2 : -1 , : ]
    data2_Y_test = data2_Y[train_n2 : -1 ]

    from sklearn.model_selection import cross_val_score

############################## dimensionality reduction ##############################

    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import FactorAnalysis

    pca1 = PCA(n_components = 2)
    data1_X_pca = pca1.fit_transform(data1_X)
    pca2 = PCA(n_components = 3)
    data2_X_pca = pca2.fit_transform(data2_X)

    ica1 = FastICA(n_components = 19)
    data1_X_ica = ica1.fit_transform(data1_X)[:, [0, 4, 5, 7, 8, 9, 10, 12, 13, 16, 18]]
    ica2 = FastICA(n_components = 3)
    data2_X_ica = ica2.fit_transform(data2_X)

############################## clustering ##############################

    from sklearn.cluster import KMeans as kmeans
    from sklearn.mixture import GaussianMixture as GMM

    clusters =  [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    temp = len(clusters)
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    score_kmean_pca1 = np.zeros(temp)
    score_gmm_pca1 = np.zeros(temp)
    score_kmean_pca2 = np.zeros(temp)
    score_gmm_pca2 = np.zeros(temp)
    score_kmean_ica1 = np.zeros(temp)
    score_gmm_ica1 = np.zeros(temp)
    score_kmean_ica2 = np.zeros(temp)
    score_gmm_ica2 = np.zeros(temp)

    for i in range(0, temp):
        km.set_params(n_clusters = clusters[i])
        gmm.set_params(n_components = clusters[i])

        km.fit(data1_X_pca)
        gmm.fit(data1_X_pca)
        score_kmean_pca1[i] = km.score(data1_X_pca)
        score_gmm_pca1[i] = gmm.score(data1_X_pca)    

        km.fit(data2_X_pca)
        gmm.fit(data2_X_pca)
        score_kmean_pca2[i] = km.score(data2_X_pca)
        score_gmm_pca2[i] = gmm.score(data2_X_pca)

        km.fit(data1_X_ica)
        gmm.fit(data1_X_ica)
        score_kmean_ica1[i] = km.score(data1_X_ica)
        score_gmm_ica1[i] = gmm.score(data1_X_ica)    

        km.fit(data2_X_ica)
        gmm.fit(data2_X_ica)
        score_kmean_ica2[i] = km.score(data2_X_ica)
        score_gmm_ica2[i] = gmm.score(data2_X_ica)

        print i+1

    file_3 = open('file_3.txt','w')

    file_3.write("Cluster")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%i" % clusters[i])
    file_3.write("\n")

    file_3.write("k-mean score pca1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_kmean_pca1[i])
    file_3.write("\n")

    file_3.write("gmm score pca1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_gmm_pca1[i])
    file_3.write("\n")

    file_3.write("k-mean score pca2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_kmean_pca2[i])
    file_3.write("\n")

    file_3.write("gmm score pca2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_gmm_pca2[i])
    file_3.write("\n")

    file_3.write("k-mean score ica1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_kmean_ica1[i])
    file_3.write("\n")

    file_3.write("gmm score ica1")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_gmm_ica1[i])
    file_3.write("\n")

    file_3.write("k-mean score ica2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_kmean_ica2[i])
    file_3.write("\n")

    file_3.write("gmm score ica2")
    for i in range(0, temp):
        file_3.write(";")
        file_3.write("%1.9f" % score_gmm_ica2[i])
    file_3.write("\n")

    file_3.close()
   
print "========== END =========="
