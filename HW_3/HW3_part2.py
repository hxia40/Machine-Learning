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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

    #
    # # data1_y_train = one_hot.fit_transform(data1_y_train.reshape(-1, 1)).todense()
    # # data1_y_test = one_hot.transform(data1_y_test.reshape(-1, 1)).todense()

    # set1_name = "ESR"
    # set1 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, 1:]
    # # set1 = np.genfromtxt('default_of_credit_card_clients.csv', delimiter=',', dtype=None)[1:6001, 1:]
    #
    # set1 = set1.astype(int)
    #
    # data1_X = set1[:, :-1]
    # data1_X = scaler.fit_transform(data1_X)
    # data1_y = set1[:, -1]
    # data1_X = scaler.fit_transform(data1_X)
    #
    # data1_X_train, data1_X_test, data1_y_train, data1_y_test = train_test_split(data1_X, data1_y, test_size=0.2,
    #                                                                             random_state=0, stratify=data1_y)

    '''Load and standardize data set ESR'''
    set2_name = "ESR"
    set2 = np.genfromtxt('Epileptic_Seizure_Recognition.csv', delimiter=',', dtype=None)[1:6001, 1:]
    # set2 = np.loadtxt(open("default_of_credit_card_clients.csv", "rb"), delimiter=",", skiprows=1)

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

    file_2 = open('part_2.txt', 'w')

    ''' PCA '''
    # pca = PCA()
    # ########### MNIST - PCA #############
    #
    # error_rate_train_1 = np.zeros(np.shape(data1_X_train)[1])
    # error_rate_test_1 = np.zeros(np.shape(data1_X_train)[1])
    #
    # DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, max_depth = None )
    #
    # error_rate_train_DT_1 = sum(
    #         DT1.fit(data1_X_train, data1_y_train).predict(data1_X_train) == data1_y_train) * 1.0 / data1_y_train.shape[0]
    # print "error_rate_train_DT_1", error_rate_train_DT_1
    # error_rate_test_DT_1 = sum(
    #         DT1.fit(data1_X_train, data1_y_train).predict(data1_X_test) == data1_y_test) * 1.0 / data1_y_test.shape[0]
    # print "error_rate_test_DT_2", error_rate_test_DT_1
    #
    # for i in range(0, np.shape(data1_X_train)[1]):
    #     print i
    #     start_time = time.time()
    #     pca.set_params(n_components=i + 1)
    #     data1_X_train_pca = pca.fit_transform(data1_X_train)
    #
    #     data1_X_test_pca = pca.transform(data1_X_test)
    #
    #     error_rate_train_1[i] = sum(
    #         DT1.fit(data1_X_train_pca, data1_y_train).predict(data1_X_train_pca) == data1_y_train) * 1.0 /data1_y_train.shape[0]
    #     print("error_rate_train_1[%f]" %i), error_rate_train_1[i]
    #     error_rate_test_1[i] = sum(
    #         DT1.fit(data1_X_train_pca, data1_y_train).predict(data1_X_test_pca) == data1_y_test) * 1.0 / data1_y_test.shape[0]
    #     print("error_rate_test_1[%f]" % i), error_rate_test_1[i]
    #     print "time consumed:", time.time()-start_time
    #
    # file_2.write("PCA_error_rate_train_1")
    # for i in range(0, len(error_rate_train_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_1[i])
    # file_2.write("\n")
    #
    # file_2.write("PCA_free_error_rate_test_1;")
    # file_2.write("%1.9f" % error_rate_train_DT_1)
    # file_2.write("\n")
    #
    # file_2.write("PCA_error_rate_test_1")
    # for i in range(0, len(error_rate_test_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_1[i])
    # file_2.write("\n")
    #
    # file_2.write("PCA_free_error_rate_test_1;")
    # file_2.write("%1.9f" % error_rate_test_DT_1)
    # file_2.write("\n")
    #
    # ########## _ESR - PCA #############
    # error_rate_train_2 = np.zeros(np.shape(data2_X_train)[1])
    # error_rate_test_2 = np.zeros(np.shape(data2_X_train)[1])
    #
    # DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=33, max_depth = None )
    # error_rate_train_DT_2 = sum(
    #         DT2.fit(data2_X_train, data2_y_train).predict(data2_X_train) == data2_y_train) * 1.0 / data2_y_train.shape[0]
    # print "error_rate_train_DT_2", error_rate_train_DT_2
    # error_rate_test_DT_2 = sum(
    #         DT2.fit(data2_X_train, data2_y_train).predict(data2_X_test) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    # print "error_rate_test_DT_2", error_rate_test_DT_2
    #
    # for i in range(0, np.shape(data2_X_train)[1]):
    #     print i
    #     start_time = time.time()
    #     pca.set_params(n_components=i + 1)
    #     data2_X_train_pca = pca.fit_transform(data2_X_train)
    #     data2_X_test_pca = pca.transform(data2_X_test)
    #
    #     error_rate_train_2[i] = sum(
    #         DT2.fit(data2_X_train_pca, data2_y_train).predict(data2_X_train_pca) == data2_y_train) * 1.0 /data2_y_train.shape[0]
    #     print("error_rate_train_2[%f]" %i), error_rate_train_2[i]
    #     error_rate_test_2[i] = sum(
    #         DT2.fit(data2_X_train_pca, data2_y_train).predict(data2_X_test_pca) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    #     print("error_rate_test_2[%f]" % i), error_rate_test_2[i]
    #     print "time consumed:", time.time() - start_time
    #
    # file_2.write("PCA_error_rate_train_2")
    # for i in range(0, len(error_rate_train_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_2[i])
    # file_2.write("\n")
    #
    # file_2.write("PCA_free_error_rate_test_2;")
    # file_2.write("%1.9f" % error_rate_train_DT_2)
    # file_2.write("\n")
    #
    # file_2.write("PCA_error_rate_test_2")
    # for i in range(0, len(error_rate_test_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_2[i])
    # file_2.write("\n")
    #
    # file_2.write("PCA_free_error_rate_test_2;")
    # file_2.write("%1.9f" % error_rate_test_DT_2)
    # file_2.write("\n")
    # #
    # #
    # #
    # # # pca = PCA(random_state=5)
    # # # pca.fit(data1_X_train).transform(data1_X_train)
    # # # pca_var_1 = pca.explained_variance_ratio_
    # # #
    # # # pca.fit(data2_X_train).transform(data2_X_train)
    # # # pca_var_2 = pca.explained_variance_ratio_
    # # #
    # # # file_2.write("PCA_variance_1")
    # # # for i in range(0, len(pca_var_1)):
    # # #     file_2.write(";")
    # # #     file_2.write("%1.9f" % pca_var_1[i])
    # # # file_2.write("\n")
    # # #
    # # # file_2.write("PCA_variance_2")
    # # # for i in range(0, len(pca_var_2)):
    # # #     file_2.write(";")
    # # #     file_2.write("%1.9f" % pca_var_2[i])
    # # # file_2.write("\n")

    ''' ICA - finding number of features '''

    ica = FastICA()
    ########## MNIST - ICA ###########
    error_rate_train_1 = np.zeros(np.shape(data1_X_train)[1])
    error_rate_test_1 = np.zeros(np.shape(data1_X_train)[1])
    kurt1_train = np.zeros(np.shape(data1_X_train)[1])
    kurt1_test = np.zeros(np.shape(data1_X_test)[1])

    DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, max_depth = None )
    # DT1 = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    # DT1 = svm.SVC(C=0.418, kernel='rbf', max_iter=-1)
    error_rate_train_DT_1 = sum(
            DT1.fit(data1_X_train, data1_y_train).predict(data1_X_train) == data1_y_train) * 1.0 / data1_y_train.shape[0]
    print "error_rate_train_DT_1", error_rate_train_DT_1
    error_rate_test_DT_1 = sum(
            DT1.fit(data1_X_train, data1_y_train).predict(data1_X_test) == data1_y_test) * 1.0 / data1_y_test.shape[0]
    print "error_rate_test_DT_2", error_rate_test_DT_1

    for i in range(0, np.shape(data1_X_train)[1]):
        print i
        start_time = time.time()
        ica.set_params(n_components=i + 1)
        data1_X_train_ica = ica.fit_transform(data1_X_train)  # data2_X_train is observation, data2_X_train_ica is ICAed
        # A_1 = ica.mixing_  # Get estimated mixing matrix
        # # print "A_2", A_2
        # data1_X_test_ica = np.dot(data1_X_test, A_1)
        data1_X_test_ica = ica.transform(data1_X_test)

        error_rate_train_1[i] = sum(
            DT1.fit(data1_X_train_ica, data1_y_train).predict(data1_X_train_ica) == data1_y_train) * 1.0 /data1_y_train.shape[0]
        print("error_rate_train_1[%f]" %i), error_rate_train_1[i]
        error_rate_test_1[i] = sum(
            DT1.fit(data1_X_train_ica, data1_y_train).predict(data1_X_test_ica) == data1_y_test) * 1.0 / data1_y_test.shape[0]
        print("error_rate_test_1[%f]" % i), error_rate_test_1[i]
        print "time consumed:", time.time()-start_time

    file_2.write("ICA_error_rate_train_1")
    for i in range(0, len(error_rate_train_1)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_train_1[i])
    file_2.write("\n")

    file_2.write("ICA_free_error_rate_test_1;")
    file_2.write("%1.9f" % error_rate_train_DT_1)
    file_2.write("\n")

    file_2.write("ICA_error_rate_test_1")
    for i in range(0, len(error_rate_test_1)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_test_1[i])
    file_2.write("\n")

    file_2.write("ICA_free_error_rate_test_1;")
    file_2.write("%1.9f" % error_rate_test_DT_1)
    file_2.write("\n")

    ########### ESR - ICA #############
    error_rate_train_2 = np.zeros(np.shape(data2_X_train)[1])
    error_rate_test_2 = np.zeros(np.shape(data2_X_train)[1])
    kurt2_train = np.zeros(np.shape(data2_X_train)[1])
    kurt2_test = np.zeros(np.shape(data2_X_test)[1])

    DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=33, max_depth = None )
    error_rate_train_DT_2 = sum(
            DT2.fit(data2_X_train, data2_y_train).predict(data2_X_train) == data2_y_train) * 1.0 / data2_y_train.shape[0]
    print "error_rate_train_DT_2", error_rate_train_DT_2
    error_rate_test_DT_2 = sum(
            DT2.fit(data2_X_train, data2_y_train).predict(data2_X_test) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    print "error_rate_test_DT_2", error_rate_test_DT_2

    for i in range(0, np.shape(data2_X_train)[1]):
        print i
        start_time = time.time()
        ica.set_params(n_components=i + 1)
        data2_X_train_ica = ica.fit_transform(data2_X_train)  # data2_X_train is observation, data2_X_train_ica is ICAed
        # A_2 = ica.mixing_  # Get estimated mixing matrix
        # # print "A_2", A_2
        # data2_X_test_ica = np.dot(data2_X_test, A_2)
        data2_X_test_ica = ica.transform(data2_X_test)

        error_rate_train_2[i] = sum(
            DT2.fit(data2_X_train_ica, data2_y_train).predict(data2_X_train_ica) == data2_y_train) * 1.0 /data2_y_train.shape[0]
        print("error_rate_train_2[%f]" %i), error_rate_train_2[i]
        error_rate_test_2[i] = sum(
            DT2.fit(data2_X_train_ica, data2_y_train).predict(data2_X_test_ica) == data2_y_test) * 1.0 / data2_y_test.shape[0]
        print("error_rate_test_2[%f]" % i), error_rate_test_2[i]
        print "time consumed:", time.time() - start_time
        # # ica.set_params(n_components=15)
        # temp2 = ica.fit_transform(data2_X_train)
        # temp2 = pd.DataFrame(temp2)
        # print "temp2", temp2
        # print "data2_X_train_ica", data2_X_train_ica
        # print "pd.Dataframe(data2_X_train_ica)", pd.Dataframe(data2_X_train_ica)
        # # print "kurt2", temp2.kurt(axis=0)
        # print "========="
        # # kurt2_train[i] = pd.Dataframe(data2_X_train_ica).kurt(axis=0)
        # # print("kurt2_train[%f]" % i), kurt2_train[i]
        # # kurt2_test[i] = pd.Dataframe(data2_X_test_ica).kurt(axis=0)
        # # print("kurt2_test[%f]" % i), kurt2_test[i]

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
    file_2.write("%1.9f" % error_rate_train_DT_2)
    file_2.write("\n")

    file_2.write("ICA_error_rate_test_2")
    for i in range(0, len(error_rate_test_2)):
        file_2.write(";")
        file_2.write("%1.9f" % error_rate_test_2[i])
    file_2.write("\n")

    file_2.write("ICA_free_error_rate_test_2;")
    file_2.write("%1.9f" % error_rate_test_DT_2)
    file_2.write("\n")


    ''' ICA - calculate kurotosis '''
    # #
    # #
    # ica.set_params(n_components=14)
    # temp1 = ica.fit_transform(data1_X_train)
    # temp1 = pd.DataFrame(temp1)
    # kurt1 = temp1.kurt(axis=0)
    #
    # ica.set_params(n_components=87)
    # temp2 = ica.fit_transform(data2_X_train)
    # temp2 = pd.DataFrame(temp2)
    # kurt2 = temp2.kurt(axis=0)
    #
    # file_2.write("ICA_kurt1_n_component=14")
    # for i in range(0, len(kurt1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % kurt1[i])
    # file_2.write("\n")
    #
    # file_2.write("ICA_kurt2_n_component=87")
    # for i in range(0, len(kurt2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % kurt2[i])
    # file_2.write("\n")

    ''' RP '''
    # grp = GaussianRandomProjection()
    # ########## MNIST - RP ###########
    #
    # error_rate_train_1 = np.zeros(np.shape(data1_X_train)[1])
    # error_rate_test_1 = np.zeros(np.shape(data1_X_train)[1])
    #
    # DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, max_depth=None)
    #
    # error_rate_train_DT_1 = sum(
    #     DT1.fit(data1_X_train, data1_y_train).predict(data1_X_train) == data1_y_train) * 1.0 / data1_y_train.shape[0]
    # print "error_rate_train_DT_1", error_rate_train_DT_1
    # error_rate_test_DT_1 = sum(
    #     DT1.fit(data1_X_train, data1_y_train).predict(data1_X_test) == data1_y_test) * 1.0 / data1_y_test.shape[0]
    # print "error_rate_test_DT_2", error_rate_test_DT_1
    #
    # for i in range(0, np.shape(data1_X_train)[1]):
    #     print i
    #     start_time = time.time()
    #     grp.set_params(n_components=i + 1)
    #     data1_X_train_grp = grp.fit_transform(data1_X_train)  # data2_X_train is observation, data2_X_train_ica is ICAed
    #     # A_1 = ica.mixing_  # Get estimated mixing matrix
    #     # # print "A_2", A_2
    #     # data1_X_test_ica = np.dot(data1_X_test, A_1)
    #     data1_X_test_grp = grp.transform(data1_X_test)
    #
    #     error_rate_train_1[i] = sum(
    #         DT1.fit(data1_X_train_grp, data1_y_train).predict(data1_X_train_grp) == data1_y_train) * 1.0 / \
    #                             data1_y_train.shape[0]
    #     print("error_rate_train_1[%f]" % i), error_rate_train_1[i]
    #     error_rate_test_1[i] = sum(
    #         DT1.fit(data1_X_train_grp, data1_y_train).predict(data1_X_test_grp) == data1_y_test) * 1.0 / \
    #                            data1_y_test.shape[0]
    #     print("error_rate_test_1[%f]" % i), error_rate_test_1[i]
    #     print "time consumed:", time.time() - start_time
    #
    # file_2.write("GRP_error_rate_train_1")
    # for i in range(0, len(error_rate_train_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_1[i])
    # file_2.write("\n")
    #
    # file_2.write("GRP_free_error_rate_test_1;")
    # file_2.write("%1.9f" % error_rate_train_DT_1)
    # file_2.write("\n")
    #
    # file_2.write("GRP_error_rate_test_1")
    # for i in range(0, len(error_rate_test_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_1[i])
    # file_2.write("\n")
    #
    # file_2.write("GRP_free_error_rate_test_1;")
    # file_2.write("%1.9f" % error_rate_test_DT_1)
    # file_2.write("\n")
    #
    # ########## ESR - RP ###########
    # error_rate_train_2 = np.zeros(np.shape(data2_X_train)[1])
    # error_rate_test_2 = np.zeros(np.shape(data2_X_train)[1])
    # kurt2_train = np.zeros(np.shape(data2_X_train)[1])
    # kurt2_test = np.zeros(np.shape(data2_X_test)[1])
    #
    # DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=33, max_depth=None)
    # error_rate_train_DT_2 = sum(
    #     DT2.fit(data2_X_train, data2_y_train).predict(data2_X_train) == data2_y_train) * 1.0 / data2_y_train.shape[0]
    # print "error_rate_train_DT_2", error_rate_train_DT_2
    # error_rate_test_DT_2 = sum(
    #     DT2.fit(data2_X_train, data2_y_train).predict(data2_X_test) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    # print "error_rate_test_DT_2", error_rate_test_DT_2
    #
    # for i in range(0, np.shape(data2_X_train)[1]):
    #     print i
    #     start_time = time.time()
    #     grp.set_params(n_components=i + 1)
    #     data2_X_train_grp = grp.fit_transform(data2_X_train)  # data2_X_train is observation, data2_X_train_ica is ICAed
    #     # A_2 = ica.mixing_  # Get estimated mixing matrix
    #     # # print "A_2", A_2
    #     # data2_X_test_ica = np.dot(data2_X_test, A_2)
    #     data2_X_test_grp = grp.transform(data2_X_test)
    #
    #     error_rate_train_2[i] = sum(
    #         DT2.fit(data2_X_train_grp, data2_y_train).predict(data2_X_train_grp) == data2_y_train) * 1.0 / \
    #                             data2_y_train.shape[0]
    #     print("error_rate_train_2[%f]" % i), error_rate_train_2[i]
    #     error_rate_test_2[i] = sum(
    #         DT2.fit(data2_X_train_grp, data2_y_train).predict(data2_X_test_grp) == data2_y_test) * 1.0 / \
    #                            data2_y_test.shape[0]
    #     print("error_rate_test_2[%f]" % i), error_rate_test_2[i]
    #     print "time consumed:", time.time() - start_time
    #
    # file_2.write("GRP_error_rate_train_2")
    # for i in range(0, len(error_rate_train_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_2[i])
    # file_2.write("\n")
    #
    # file_2.write("GRP_free_error_rate_test_2;")
    # file_2.write("%1.9f" % error_rate_train_DT_2)
    # file_2.write("\n")
    #
    # file_2.write("GRP_error_rate_test_2")
    # for i in range(0, len(error_rate_test_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_2[i])
    # file_2.write("\n")
    #
    # file_2.write("GRP_free_error_rate_test_2;")
    # file_2.write("%1.9f" % error_rate_test_DT_2)
    # file_2.write("\n")

    ''' FA '''
    # fa = FeatureAgglomeration()
    # ########### MNIST - FA #############
    #
    # error_rate_train_1 = np.zeros(np.shape(data1_X_train)[1])
    # error_rate_test_1 = np.zeros(np.shape(data1_X_train)[1])
    #
    # DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, max_depth=None)
    #
    # error_rate_train_DT_1 = sum(
    #     DT1.fit(data1_X_train, data1_y_train).predict(data1_X_train) == data1_y_train) * 1.0 / data1_y_train.shape[0]
    # print "error_rate_train_DT_1", error_rate_train_DT_1
    # error_rate_test_DT_1 = sum(
    #     DT1.fit(data1_X_train, data1_y_train).predict(data1_X_test) == data1_y_test) * 1.0 / data1_y_test.shape[0]
    # print "error_rate_test_DT_2", error_rate_test_DT_1
    #
    # for i in range(0, np.shape(data1_X_train)[1]):
    #     print i
    #     start_time = time.time()
    #     fa.set_params(n_clusters=i + 1)
    #     data1_X_train_fa = fa.fit_transform(data1_X_train)
    #     data1_X_test_fa = fa.transform(data1_X_test)
    #
    #     error_rate_train_1[i] = sum(
    #         DT1.fit(data1_X_train_fa, data1_y_train).predict(data1_X_train_fa) == data1_y_train) * 1.0 / \
    #                             data1_y_train.shape[0]
    #     print("error_rate_train_1[%f]" % i), error_rate_train_1[i]
    #     error_rate_test_1[i] = sum(
    #         DT1.fit(data1_X_train_fa, data1_y_train).predict(data1_X_test_fa) == data1_y_test) * 1.0 / \
    #                            data1_y_test.shape[0]
    #     print("error_rate_test_1[%f]" % i), error_rate_test_1[i]
    #     print "time consumed:", time.time() - start_time
    #
    # file_2.write("FA_error_rate_train_1")
    # for i in range(0, len(error_rate_train_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_1[i])
    # file_2.write("\n")
    #
    # file_2.write("FA_free_error_rate_test_1;")
    # file_2.write("%1.9f" % error_rate_train_DT_1)
    # file_2.write("\n")
    #
    # file_2.write("FA_error_rate_test_1")
    # for i in range(0, len(error_rate_test_1)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_1[i])
    # file_2.write("\n")
    #
    # file_2.write("FA_free_error_rate_test_1;")
    # file_2.write("%1.9f" % error_rate_test_DT_1)
    # file_2.write("\n")
    #
    # ########### _ESR - FA #############
    # error_rate_train_2 = np.zeros(np.shape(data2_X_train)[1])
    # error_rate_test_2 = np.zeros(np.shape(data2_X_train)[1])
    #
    # DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=33, max_depth=None)
    # error_rate_train_DT_2 = sum(
    #     DT2.fit(data2_X_train, data2_y_train).predict(data2_X_train) == data2_y_train) * 1.0 / data2_y_train.shape[0]
    # print "error_rate_train_DT_2", error_rate_train_DT_2
    # error_rate_test_DT_2 = sum(
    #     DT2.fit(data2_X_train, data2_y_train).predict(data2_X_test) == data2_y_test) * 1.0 / data2_y_test.shape[0]
    # print "error_rate_test_DT_2", error_rate_test_DT_2
    #
    # for i in range(0, np.shape(data2_X_train)[1]):
    #     print i
    #     start_time = time.time()
    #     fa.set_params(n_clusters=i + 1)
    #     data2_X_train_fa = fa.fit_transform(data2_X_train)  # data2_X_train is observation, data2_X_train_ica is ICAed
    #     data2_X_test_fa = fa.transform(data2_X_test)
    #
    #     error_rate_train_2[i] = sum(
    #         DT2.fit(data2_X_train_fa, data2_y_train).predict(data2_X_train_fa) == data2_y_train) * 1.0 / \
    #                             data2_y_train.shape[0]
    #     print("error_rate_train_2[%f]" % i), error_rate_train_2[i]
    #     error_rate_test_2[i] = sum(
    #         DT2.fit(data2_X_train_fa, data2_y_train).predict(data2_X_test_fa) == data2_y_test) * 1.0 / \
    #                            data2_y_test.shape[0]
    #     print("error_rate_test_2[%f]" % i), error_rate_test_2[i]
    #     print "time consumed:", time.time() - start_time
    #
    # file_2.write("FA_error_rate_train_2")
    # for i in range(0, len(error_rate_train_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_train_2[i])
    # file_2.write("\n")
    #
    # file_2.write("FA_free_error_rate_test_2;")
    # file_2.write("%1.9f" % error_rate_train_DT_2)
    # file_2.write("\n")
    #
    # file_2.write("FA_error_rate_test_2")
    # for i in range(0, len(error_rate_test_2)):
    #     file_2.write(";")
    #     file_2.write("%1.9f" % error_rate_test_2[i])
    # file_2.write("\n")
    #
    # file_2.write("FA_free_error_rate_test_2;")
    # file_2.write("%1.9f" % error_rate_test_DT_2)
    # file_2.write("\n")


print "========== END =========="
