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
# import yellowbrick

if __name__=="__main__":
    # Data Loading & Preprocessing
    scaler = preprocessing.StandardScaler()
    one_hot = preprocessing.OneHotEncoder()

    '''Load and standardize data set MNIST'''
    # set1_name = "mnist"
    #
    # train = np.genfromtxt('fashion-mnist_train_minor.csv', delimiter=',')[1:, :]
    # test = np.genfromtxt('fashion-mnist_test_minor.csv', delimiter=',')[1:, :]
    #
    # data1_X_train = train[:, 1:]
    # data1_y_train = train[:, 0]
    # data1_X_test = test[:, 1:]
    # data1_y_test = test[:, 0]
    #
    # data1_X_train = scaler.fit_transform(data1_X_train)
    # data1_X_test = scaler.transform(data1_X_test)
    #
    # # data1_y_train = one_hot.fit_transform(data1_y_train.reshape(-1, 1)).todense()
    # # data1_y_test = one_hot.transform(data1_y_test.reshape(-1, 1)).todense()
    set1_name = "ESR"
    set1 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, 1:]
    # set1 = np.genfromtxt('default_of_credit_card_clients.csv', delimiter=',', dtype=None)[1:6001, 1:]

    set1 = set1.astype(int)

    data1_X = set1[:, :-1]
    data1_X = scaler.fit_transform(data1_X)
    data1_y = set1[:, -1]
    data1_X = scaler.fit_transform(data1_X)

    data1_X_train, data1_X_test, data1_y_train, data1_y_test = train_test_split(data1_X, data1_y, test_size=0.2,
                                                                                random_state=0, stratify=data1_y)



    '''Load and standardize data set ESR'''
    set2_name = "ESR"
    # set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, 1:]
    set2 = np.loadtxt(open("default_of_credit_card_clients.csv", "rb"), delimiter=",", skiprows=1)

    set2 = set2.astype(int)

    data2_X = set2[:, :-1]
    data2_X = scaler.fit_transform(data2_X)
    data2_y = set2[:, -1]
    data2_X = scaler.fit_transform(data2_X)
    data2_X_train, data2_X_test, data2_y_train, data2_y_test = train_test_split(data2_X, data2_y, test_size=0.2, random_state=0, stratify=data2_y)

    # data2_y_train = one_hot.fit_transform(data2_y_train.reshape(-1, 1)).todense()
    # data2_y_test = one_hot.transform(data2_y_test.reshape(-1, 1)).todense()

    '''Load and standardize data set Bank'''
    # set3_name = "Bank"
    #
    # set3 = np.genfromtxt('default_of_credit_card_clients.csv', delimiter=',', dtype=None)[1:6001, 1:]
    #
    # set3 = set3.astype(int)
    #
    # data3_X = set3[:, :-1]
    # data3_X = scaler.fit_transform(data3_X)
    # data3_y = set3[:, -1]
    # data3_X_train, data3_X_test, data3_y_train, data3_y_test = train_test_split(data3_X, data3_y, test_size=0.2,
    #                                                                             random_state=0, stratify=data3_y)
    #
    # data3_X_train = scaler.fit_transform(data3_X_train)
    # data3_X_test = scaler.transform(data3_X_test)
    #
    # # data2_y_train = one_hot.fit_transform(data2_y_train.reshape(-1, 1)).todense()
    # # data2_y_test = one_hot.transform(data2_y_test.reshape(-1, 1)).todense()

    ############################## dimensionality reduction ##############################

    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import FactorAnalysis

    from sklearn import svm
    from sklearn import tree

    from sklearn.metrics.pairwise import pairwise_distances


    def pairwiseDistCorr(X1, X2):
        assert X1.shape[0] == X2.shape[0]

        d1 = pairwise_distances(X1)
        d2 = pairwise_distances(X2)
        return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


    file_2 = open('file_2.txt', 'w')

    # ############################## PCA ##############################
    #
    # pca = PCA(random_state=5)
    # pca.fit(data1_X_train)
    # pca_var_1 = pca.explained_variance_ratio_
    # pca_sing_1 = pca.singular_values_
    #
    # pca.fit(data2_X_train)
    # pca_var_2 = pca.explained_variance_ratio_
    # pca_sing_2 = pca.singular_values_
    #
    # file_2.write("PCA_variance_1")
    # for i in range(0, len(pca_var_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % pca_var_1[i])
    # file_2.write("\n")
    #
    # file_2.write("PCA_singular_1")
    # for i in range(0, len(pca_sing_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % pca_sing_1[i])
    # file_2.write("\n")
    #
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

    ############################## ICA - finding number of features ##############################

    ica = FastICA(random_state=5)
    # error_rate_train_1 = np.zeros(np.shape(data1_X_train)[1])
    # error_rate_test_1 = np.zeros(np.shape(data1_X_train)[1])
    # for i in range(0, np.shape(data1_X_train)[1]):
    #     print i
    #     ica.set_params(n_components=i + 1)
    #     DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=0.005)
    #     error_rate_train_1[i] = sum(
    #         DT1.fit(ica.fit_transform(data1_X_train), data1_y_train).predict(ica.fit_transform(data1_X_train)) == data1_y_train) * 1.0 /data1_y_train.shape[0]
    #     error_rate_test_1[i] = sum(
    #         DT1.fit(ica.fit_transform(data1_X_train), data1_y_train).predict(ica.fit_transform(data1_X_test)) == data1_y_test) * 1.0 /data1_y_test.shape[0]

    error_rate_train_2 = np.zeros(np.shape(data2_X_train)[1])
    error_rate_test_2 = np.zeros(np.shape(data2_X_train)[1])
    DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=0.005)
    for i in range(0, np.shape(data2_X_train)[1]):
        print i
        ica.set_params(n_components=i + 1)

        error_rate_train_2[i] = sum(
            DT2.fit(ica.fit_transform(data2_X_train), data2_y_train).predict(ica.fit_transform(data2_X_train)) == data2_y_train) * 1.0 /data2_y_train.shape[0]
        error_rate_test_2[i] = sum(
            DT2.fit(ica.fit_transform(data2_X_train), data2_y_train).predict(ica.fit_transform(data2_X_test)) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    error_rate_train_DT = sum(
            DT2.fit(data2_X_train, data2_y_train).predict(data2_X_train) == data2_y_train) * 1.0 / data2_y_train.shape[0]
    error_rate_test_DT = sum(
            DT2.fit(data2_X_train, data2_y_train).predict(data2_X_test) == data2_y_test) * 1.0 / data2_y_test.shape[0]


    # ica.set_params(n_components=15)
    # print "component = 15"
    # print sum(DT2.fit(ica.fit_transform(data2_X_train), data2_y_train).predict(ica.fit_transform(data2_X_test)) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    # ica.set_params(n_components=16)
    # print "component = 16"
    # print sum(DT2.fit(ica.fit_transform(data2_X_train), data2_y_train).predict(
    #     ica.fit_transform(data2_X_test)) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    # ica.set_params(n_components=17)
    # print "component = 17"
    # print sum(DT2.fit(ica.fit_transform(data2_X_train), data2_y_train).predict(
    #     ica.fit_transform(data2_X_test)) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    #
    # file_2.write("ICA_error_rate_train_1")
    # for i in range(0, len(error_rate_train_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_1[i])
    # file_2.write("\n")
    #
    # file_2.write("ICA_error_rate_test_1")
    # for i in range(0, len(error_rate_test_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_1[i])
    # file_2.write("\n")

    file_2.write("ICA_error_rate_train_2")
    for i in range(0, len(error_rate_train_2)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_train_2[i])
    file_2.write("\n")

    file_2.write("ICA_free_error_rate_test_2;")
    file_2.write("%1.9f" % error_rate_train_DT)
    file_2.write("\n")

    file_2.write("ICA_error_rate_test_2")
    for i in range(0, len(error_rate_test_2)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_test_2[i])
    file_2.write("\n")

    file_2.write("ICA_free_error_rate_test_2;")
    file_2.write("%1.9f" % error_rate_test_DT)
    file_2.write("\n")

    # ############################## ICA - calculate kurotosis ##############################
    #
    # i1 = np.argmax(error_rate_1) + 1
    # ica.set_params(n_components=i1)
    # temp1 = ica.fit_transform(data1_X_train)
    # temp1 = pd.DataFrame(temp1)
    # kurt1 = temp1.kurt(axis=0)
    #
    ica.set_params(n_components=15)
    temp2 = ica.fit_transform(data2_X_train)
    temp2 = pd.DataFrame(temp2)
    kurt2 = temp2.kurt(axis=0)

    # file_2.write("ICA_kurt1")
    # for i in range(0, len(kurt1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % kurt1[i])
    # file_2.write("\n")
    #
    file_2.write("ICA_kurt2_n_component=15")
    for i in range(0, len(kurt2)):
        file_2.write(";")
        file_2.write("%1.9f" % kurt2[i])
    file_2.write("\n")

    # ############################## RP ##############################
    #
    # grp = GaussianRandomProjection(random_state=5)
    # error_rate_1 = np.zeros(np.shape(data1_X_train)[1])
    # for i in range(0, np.shape(data1_X_train)[1]):
    #     grp.set_params(n_components=i + 1)
    #     DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=0.005)
    #     error_rate_1[i] = sum(
    #         DT1.fit(grp.fit_transform(data1_X_train), data1_y_train).predict(grp.fit_transform(data1_X_train)) == data1_y_train) * 1.0
    #     print i + 1
    # i1 = np.argmax(error_rate_1) + 1
    # grp.set_params(n_components=i1)
    # recon1 = range(0, 2)  # pairwiseDistCorr(grp.fit_transform(data1_X), data1_X)
    #
    # error_rate_2 = np.zeros(np.shape(data2_X_train)[1])
    # for i in range(0, np.shape(data2_X_train)[1]):
    #     grp.set_params(n_components=i + 1)
    #     DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=0.005)
    #     error_rate_2[i] = sum(
    #         DT2.fit(grp.fit_transform(data2_X_train), data2_y_train).predict(grp.fit_transform(data2_X_train)) == data2_y_train) * 1.0
    # i2 = np.argmax(error_rate_2) + 1
    # grp.set_params(n_components=i2)
    # recon2 = range(0, 2)  # pairwiseDistCorr(grp.fit_transform(data2_X), data2_X)
    #
    # file_2.write("RP_error_rate_1")
    # for i in range(0, len(error_rate_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_1[i])
    # file_2.write("\n")
    #
    # file_2.write("RP_no_component_1")
    # file_2.write(";")
    # file_2.write("%i" % i1)
    # file_2.write("\n")
    #
    # file_2.write("RP_recon1")
    # for i in range(0, len(recon1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % recon1[i])
    # file_2.write("\n")
    #
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
    #
    # ############################## FA ##############################
    #
    # fa = FactorAnalysis()
    # fa.fit(data1_X_train)
    # # fa_var_1 = fa.components_
    # fa_noise_1 = fa.noise_variance_
    #
    # fa.fit(data2_X_train)
    # # fa_var_2 = fa.components_
    # fa_noise_2 = fa.noise_variance_
    #
    # # file_2.write("FA_variance_1")
    # # for i in range(0, len(pca_var_1)):
    # # file_2.write(";")
    # # file_2.write("%1.9f" % fa_var_1[i])
    # # file_2.write("\n")
    #
    # file_2.write("FA_noise_1")
    # for i in range(0, len(fa_noise_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % fa_noise_1[i])
    # file_2.write("\n")
    #
    # # file_2.write("FA_variance_2")
    # # for i in range(0, len(fa_var_2)):
    # # file_2.write(";")
    # # file_2.write("%1.9f" % fa_var_2[i])
    # # file_2.write("\n")
    #
    # file_2.write("FA_noise_2")
    # for i in range(0, len(fa_noise_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % fa_noise_2[i])
    # file_2.write("\n")
    #
    # file_2.close()

print "========== END =========="
