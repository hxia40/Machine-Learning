import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy as sp
import time

if __name__=="__main__":

    # Data Loading & Preprocessing

    data1 = np.loadtxt(open("default_of_credit_card_clients.csv", "rb"), delimiter=",", skiprows=1)
    # data2 = np.loadtxt(open("FFdFactorsPs.csv", "rb"), delimiter=",", skiprows=1)

    data1_X = data1[: , 0:-1]
    data1_Y = data1[: , -1]

    # start_date = np.min(np.argwhere(data2[: , 0] == 19700102))
    # data2_X = np.column_stack((data2[ (start_date) : -1 , (2, 3)], data2[ (start_date - 1) : -2 , 1]))
    # data2_Y = data2[ (start_date) : -1 , 1]
    # data2_Y = (data2_Y >= 0) * 1

    n1 = data1_Y.shape[0]
    # n2 = data2_Y.shape[0]
    train_n1 = int(round(n1 * 0.6, 0))
    # train_n2 = int(round(n2 * 0.6, 0))
    test_n1 = n1 - train_n1
    # test_n2 = n2 - train_n2

    data1_X_train = data1_X[0:train_n1, : ]
    data1_Y_train = data1_Y[0:train_n1]
    data1_X_test = data1_X[train_n1 : -1 , : ]
    data1_Y_test = data1_Y[train_n1 : -1 ]

    # data2_X_train = data2_X[0:train_n2, : ]
    # data2_Y_train = data2_Y[0:train_n2]
    # data2_X_test = data2_X[train_n2 : -1 , : ]
    # data2_Y_test = data2_Y[train_n2 : -1 ]

    from sklearn.model_selection import cross_val_score

############################## dimensionality reduction ##############################

    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import FactorAnalysis

    from sklearn import svm
    from sklearn import tree

    from sklearn.metrics.pairwise import pairwise_distances
    def pairwiseDistCorr(X1,X2):
        assert X1.shape[0] == X2.shape[0]

        d1 = pairwise_distances(X1)
        d2 = pairwise_distances(X2)
        return np.corrcoef(d1.ravel(),d2.ravel())[0,1]


    file_2 = open('file_2.txt','w')

############################## PCA ##############################

    pca = PCA(random_state=5)
    pca.fit(data1_X)
    pca_var_1 = pca.explained_variance_ratio_
    pca_sing_1 = pca.singular_values_

    # pca.fit(data2_X)
    # pca_var_2 = pca.explained_variance_ratio_
    # pca_sing_2 = pca.singular_values_

    file_2.write("PCA_variance_1")
    for i in range(0, len(pca_var_1)):
        file_2.write(";")
        file_2.write("%1.9f" % pca_var_1[i])
    file_2.write("\n")

    file_2.write("PCA_singular_1")
    for i in range(0, len(pca_sing_1)):
        file_2.write(";")
        file_2.write("%1.9f" % pca_sing_1[i])
    file_2.write("\n")

    # file_2.write("PCA_variance_2")
    # for i in range(0, len(pca_var_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % pca_var_2[i])
    # file_2.write("\n")
    #
    # file_2.write("PCA_singular_2")
    # for i in range(0, len(pca_sing_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % pca_sing_2[i])
    # file_2.write("\n")

############################## ICA ##############################

    ica = FastICA(random_state=5)
    error_rate_1 = np.zeros(np.shape(data1_X)[1])
    for i in range(0, np.shape(data1_X)[1]):
        ica.set_params(n_components = i+1)
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
        error_rate_1[i] = sum(DT1.fit(ica.fit_transform(data1_X), data1_Y).predict(ica.fit_transform(data1_X)) == data1_Y) * 1.0 / data1_Y.shape[0]

        print i+1
    i1 = np.argmax(error_rate_1) + 1
    ica.set_params(n_components = i1)
    temp1 = ica.fit_transform(data1_X)
    temp1 = pd.DataFrame(temp1)
    kurt1 = temp1.kurt(axis=0)

    # error_rate_2 = np.zeros(np.shape(data2_X)[1])
    # for i in range(0, np.shape(data2_X)[1]):
    #     ica.set_params(n_components = i+1)
    #     DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
    #     error_rate_2[i] = sum(DT2.fit(ica.fit_transform(data2_X), data2_Y).predict(ica.fit_transform(data2_X)) == data2_Y) * 1.0 / n2
    # i2 = np.argmax(error_rate_2) + 1
    # ica.set_params(n_components = i2)
    # temp2 = ica.fit_transform(data2_X)
    # temp2 = pd.DataFrame(temp2)
    # kurt2 = temp2.kurt(axis=0)

    file_2.write("ICA_error_rate_1")
    for i in range(0, len(error_rate_1)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_1[i])
    file_2.write("\n")

    file_2.write("ICA_no_component_1")
    file_2.write(";")
    file_2.write("%i" % i1)
    file_2.write("\n")

    file_2.write("ICA_kurt1")
    for i in range(0, len(kurt1)):
        file_2.write(";")
        file_2.write("%1.9f" % kurt1[i])
    file_2.write("\n")

    # file_2.write("ICA_error_rate_2")
    # for i in range(0, len(error_rate_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_2[i])
    # file_2.write("\n")
    #
    # file_2.write("ICA_no_component_2")
    # file_2.write(";")
    # file_2.write("%i" % i2)
    # file_2.write("\n")
    #
    # file_2.write("ICA_kurt2")
    # for i in range(0, len(kurt2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % kurt2[i])
    # file_2.write("\n")

############################## RP ##############################

    grp = GaussianRandomProjection(random_state=5)
    error_rate_1 = np.zeros(np.shape(data1_X)[1])
    for i in range(0, np.shape(data1_X)[1]):
        grp.set_params(n_components = i+1)
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
        error_rate_1[i] = sum(DT1.fit(grp.fit_transform(data1_X), data1_Y).predict(grp.fit_transform(data1_X)) == data1_Y) * 1.0 / n1
        print i+1
    i1 = np.argmax(error_rate_1) + 1
    grp.set_params(n_components = i1)
    recon1 = range(0, 2)#pairwiseDistCorr(grp.fit_transform(data1_X), data1_X)

    # error_rate_2 = np.zeros(np.shape(data2_X)[1])
    # for i in range(0, np.shape(data2_X)[1]):
    #     grp.set_params(n_components = i+1)
    #     DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
    #     error_rate_2[i] = sum(DT2.fit(grp.fit_transform(data2_X), data2_Y).predict(grp.fit_transform(data2_X)) == data2_Y) * 1.0 / n2
    # i2 = np.argmax(error_rate_2) + 1
    # grp.set_params(n_components = i2)
    # recon2 = range(0, 2)#pairwiseDistCorr(grp.fit_transform(data2_X), data2_X)

    file_2.write("RP_error_rate_1")
    for i in range(0, len(error_rate_1)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_1[i])
    file_2.write("\n")

    file_2.write("RP_no_component_1")
    file_2.write(";")
    file_2.write("%i" % i1)
    file_2.write("\n")

    file_2.write("RP_recon1")
    for i in range(0, len(recon1)):
        file_2.write(";")
        file_2.write("%1.9f" % recon1[i])
    file_2.write("\n")

    # file_2.write("RP_error_rate_2")
    # for i in range(0, len(error_rate_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_2[i])
    # file_2.write("\n")
    #
    # file_2.write("RP_no_component_2")
    # file_2.write(";")
    # file_2.write("%i" % i2)
    # file_2.write("\n")
    #
    # file_2.write("RP_recon2")
    # for i in range(0, len(recon2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % recon2[i])
    # file_2.write("\n")

############################## FA ##############################

    fa = FactorAnalysis()
    fa.fit(data1_X)
    #fa_var_1 = fa.components_
    fa_noise_1 = fa.noise_variance_

    # fa.fit(data2_X)
    # #fa_var_2 = fa.components_
    # fa_noise_2 = fa.noise_variance_

    #file_2.write("FA_variance_1")
    #for i in range(0, len(pca_var_1)):
        #file_2.write(";")
        #file_2.write("%1.9f" % fa_var_1[i])
    #file_2.write("\n")

    file_2.write("FA_noise_1")
    for i in range(0, len(fa_noise_1)):
        file_2.write(";")
        file_2.write("%1.9f" % fa_noise_1[i])
    file_2.write("\n")

    # #file_2.write("FA_variance_2")
    # #for i in range(0, len(fa_var_2)):
    #     #file_2.write(";")
    #     #file_2.write("%1.9f" % fa_var_2[i])
    # #file_2.write("\n")
    #
    # file_2.write("FA_noise_2")
    # for i in range(0, len(fa_noise_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % fa_noise_2[i])
    # file_2.write("\n")



    file_2.close()

print "========== END =========="
