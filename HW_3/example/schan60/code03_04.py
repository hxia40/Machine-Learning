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
    ica1 = FastICA(n_components = 19)
    data1_X_ica = ica1.fit_transform(data1_X)[:, [0, 4, 5, 7, 8, 9, 10, 12, 13, 16, 18]]
    grp1 = GaussianRandomProjection(n_components = 16)
    data1_X_rp = grp1.fit_transform(data1_X)

############################## neural network ##############################

    from sklearn.neural_network import MLPClassifier
    file_4 = open('file_4.txt','w')

    print "BASE started"
    start_time_1_base = time.time()
    ANN_base = MLPClassifier(hidden_layer_sizes=([8, 8]), random_state=1, max_iter=500)
    scores_test_ANN_base = cross_val_score(ANN_base, data1_X, data1_Y, cv = 10, scoring = 'accuracy').mean()
    elasped_time_1_base = time.time() - start_time_1_base
    print "BASE ended"

    print "PCA started"
    start_time_1_pca = time.time()
    ANN_pca = MLPClassifier(hidden_layer_sizes=([8, 8]), random_state=1, max_iter=500)
    scores_test_ANN_pca = cross_val_score(ANN_pca, data1_X_pca, data1_Y, cv = 10, scoring = 'accuracy').mean()
    elasped_time_1_pca = time.time() - start_time_1_pca
    print "PCA ended"

    print "ICA started"
    start_time_1_ica = time.time()
    ANN_ica = MLPClassifier(hidden_layer_sizes=([8, 8]), random_state=1, max_iter=500)
    scores_test_ANN_ica = cross_val_score(ANN_ica, data1_X_ica, data1_Y, cv = 10, scoring = 'accuracy').mean()
    elasped_time_1_ica = time.time() - start_time_1_ica
    print "ICE ended"

    print "RP started"
    start_time_1_rp = time.time()
    ANN_rp = MLPClassifier(hidden_layer_sizes=([8, 8]), random_state=1, max_iter=500)
    scores_test_ANN_rp = cross_val_score(ANN_rp, data1_X_rp, data1_Y, cv = 10, scoring = 'accuracy').mean()
    elasped_time_1_rp = time.time() - start_time_1_rp
    print "RP ended"

    file_4.write("no of variable (base)" + ";")
    file_4.write("%i" % data1_X.shape[1])
    file_4.write("\n")
    file_4.write("scores_test_ANN_base" + ";")
    file_4.write("%1.9f" % scores_test_ANN_base)
    file_4.write("\n")
    file_4.write("elasped_time_1" + ";")
    file_4.write("%1.9f" % elasped_time_1_base)
    file_4.write("\n")
    file_4.write("no of variable (PCA)" + ";")
    file_4.write("%i" % data1_X_pca.shape[1])
    file_4.write("\n")
    file_4.write("scores_test_ANN_PCA" + ";")
    file_4.write("%1.9f" % scores_test_ANN_pca)
    file_4.write("\n")
    file_4.write("elasped_time_1" + ";")
    file_4.write("%1.9f" % elasped_time_1_pca)
    file_4.write("\n")
    file_4.write("no of variable (ICA)" + ";")
    file_4.write("%i" % data1_X_ica.shape[1])
    file_4.write("\n")
    file_4.write("scores_test_ANN_ICA" + ";")
    file_4.write("%1.9f" % scores_test_ANN_ica)
    file_4.write("\n")
    file_4.write("elasped_time_1" + ";")
    file_4.write("%1.9f" % elasped_time_1_ica)
    file_4.write("\n")
    file_4.write("no of variable (RP)" + ";")
    file_4.write("%i" % data1_X_rp.shape[1])
    file_4.write("\n")
    file_4.write("scores_test_ANN_RP" + ";")
    file_4.write("%1.9f" % scores_test_ANN_rp)
    file_4.write("\n")
    file_4.write("elasped_time_1" + ";")
    file_4.write("%1.9f" % elasped_time_1_rp)
    file_4.write("\n")

    file_4.close()
   
print "========== END =========="
