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

############################## clustering ##############################

    from sklearn.cluster import KMeans as kmeans
    from sklearn.mixture import GaussianMixture as GMM

    clusters =  [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    temp = len(clusters)
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    score_kmean_1 = np.zeros(temp)
    score_gmm_1 = np.zeros(temp)
    score_kmean_2 = np.zeros(temp)
    score_gmm_2 = np.zeros(temp)

    for i in range(0, temp):
        km.set_params(n_clusters = clusters[i])
        gmm.set_params(n_components = clusters[i])

        km.fit(data1_X)
        gmm.fit(data1_X)
        score_kmean_1[i] = km.score(data1_X)
        score_gmm_1[i] = gmm.score(data1_X)    

        km.fit(data2_X)
        gmm.fit(data2_X)
        score_kmean_2[i] = km.score(data2_X)
        score_gmm_2[i] = gmm.score(data2_X)

        print i+1

    file_1 = open('file_1.txt','w')

    file_1.write("Cluster")
    for i in range(0, temp):
        file_1.write(";")
        file_1.write("%i" % clusters[i])
    file_1.write("\n")

    file_1.write("k-mean score 1")
    for i in range(0, temp):
        file_1.write(";")
        file_1.write("%1.9f" % score_kmean_1[i])
    file_1.write("\n")

    file_1.write("gmm score 1")
    for i in range(0, temp):
        file_1.write(";")
        file_1.write("%1.9f" % score_gmm_1[i])
    file_1.write("\n")

    file_1.write("k-mean score 2")
    for i in range(0, temp):
        file_1.write(";")
        file_1.write("%1.9f" % score_kmean_2[i])
    file_1.write("\n")

    file_1.write("gmm score 2")
    for i in range(0, temp):
        file_1.write(";")
        file_1.write("%1.9f" % score_gmm_2[i])
    file_1.write("\n")

    file_1.close()
   
print "========== END =========="
