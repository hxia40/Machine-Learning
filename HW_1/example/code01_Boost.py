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

    data1_X = data1[: , 0:11]
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

############################## Boosting ##############################

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import tree
    BOOST_file = open('number_BOOST.txt','w')

    # learning curve
    temp = 50
    scores_train_learning_BOOST1 = np.zeros(temp)
    scores_test_learning_BOOST1 = np.zeros(temp)
    scores_train_learning_BOOST2 = np.zeros(temp)
    scores_test_learning_BOOST2 = np.zeros(temp)
    temp_n1 = np.zeros(temp)
    temp_n2 = np.zeros(temp)
    for i in range(0, temp):
        temp_n1_ = int(round(train_n1 * (i+1)/temp, 0))
        temp_n2_ = int(round(train_n2 * (i+1)/temp, 0))
        temp_n1[i] = temp_n1_
        temp_n2[i] = temp_n2_
        data1_X_temp = data1_X[0:temp_n1_, : ]
        data1_Y_temp = data1_Y[0:temp_n1_]
        data2_X_temp = data2_X[0:temp_n2_, : ]
        data2_Y_temp = data2_Y[0:temp_n2_]
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
        BOOST1 = AdaBoostClassifier(base_estimator=DT1, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        scores_train_learning_BOOST1[i] = sum(BOOST1.fit(data1_X_temp, data1_Y_temp).predict(data1_X_temp) == data1_Y_temp) * 1.0 / temp_n1_
        scores_test_learning_BOOST1[i] = cross_val_score(BOOST1, data1_X_temp, data1_Y_temp, cv = 10, scoring = 'accuracy').mean()
        DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
        BOOST2 = AdaBoostClassifier(base_estimator=DT2, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        scores_train_learning_BOOST2[i] = sum(BOOST2.fit(data2_X_temp, data2_Y_temp).predict(data2_X_temp) == data2_Y_temp) * 1.0 / temp_n2_
        scores_test_learning_BOOST2[i] = cross_val_score(BOOST2, data2_X_temp, data2_Y_temp, cv = 10, scoring = 'accuracy').mean()
        print "Boosting (DT) (learning curve):", (i+1)*100.0/temp , "% done."

    fig_BOOST_1 = plt.figure()
    graph_BOOST_1 = fig_BOOST_1.add_subplot(111, )
    graph_BOOST_1.plot(temp_n1, scores_train_learning_BOOST1, label='Training Set Accuracy')
    graph_BOOST_1.plot(temp_n1, scores_test_learning_BOOST1, label='Testing Set Accuracy')
    graph_BOOST_1.set_xlabel("Sample Size")
    graph_BOOST_1.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Boosting (DT) (Dataset 1)")
    plt.legend(loc='upper right');
    plt.savefig('graph_BOOST_1.png')

    BOOST_file.write("scores_train_learning_BOOST1")
    for i in range(len(scores_train_learning_BOOST1)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_train_learning_BOOST1[i])
    BOOST_file.write("\n")
    BOOST_file.write("scores_test_learning_BOOST1")
    for i in range(len(scores_test_learning_BOOST1)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_test_learning_BOOST1[i])
    BOOST_file.write("\n")

    fig_BOOST_2 = plt.figure()
    graph_BOOST_2 = fig_BOOST_2.add_subplot(111, )
    graph_BOOST_2.plot(temp_n2, scores_train_learning_BOOST2, label='Training Set Accuracy')
    graph_BOOST_2.plot(temp_n2, scores_test_learning_BOOST2, label='Testing Set Accuracy')
    graph_BOOST_2.set_xlabel("Sample Size")
    graph_BOOST_2.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Boosting (DT) (Dataset 2)")
    plt.legend(loc='upper right');
    plt.savefig('graph_BOOST_2.png')

    BOOST_file.write("scores_train_learning_BOOST2")
    for i in range(len(scores_train_learning_BOOST2)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_train_learning_BOOST2[i])
    BOOST_file.write("\n")
    BOOST_file.write("scores_test_learning_BOOST2")
    for i in range(len(scores_test_learning_BOOST2)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_test_learning_BOOST2[i])
    BOOST_file.write("\n")

    # decision of leaf size (pruning)
    temp = 50
    scores_pruning_BOOST1 = np.zeros(temp)
    scores_pruning_BOOST2 = np.zeros(temp)
    temp_n = np.zeros(temp)
    for i in range(0, temp):
        temp_n[i] = (i+1) * 0.25 / temp
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[i])
        BOOST1 = AdaBoostClassifier(base_estimator=DT1, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        scores_pruning_BOOST1[i] = cross_val_score(BOOST1, data1_X_train, data1_Y_train, cv = 10, scoring = 'accuracy').mean()
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[i])
        BOOST2 = AdaBoostClassifier(base_estimator=DT2, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        scores_pruning_BOOST2[i] = cross_val_score(BOOST2, data2_X_train, data2_Y_train, cv = 10, scoring = 'accuracy').mean()
        print "Boosting (DT) (pruning analysis):", (i+1)*100.0/temp , "% done."

    fig_BOOST_3 = plt.figure()
    graph_BOOST_3 = fig_BOOST_3.add_subplot(111, )
    graph_BOOST_3.plot(temp_n, scores_pruning_BOOST1, label='Accuracy (Dataset 1)')
    graph_BOOST_3.plot(temp_n, scores_pruning_BOOST2, label='Accuracy (Dataset 2)')
    graph_BOOST_3.set_xlabel("Min Leaf Size (% of no. of sample)")
    graph_BOOST_3.set_ylabel("Accuracy")
    plt.title("Accuracy (different minimum leaf size) - Boosting (DT)")
    plt.legend(loc='upper right');
    plt.savefig('graph_BOOST_3.png')

    BOOST_file.write("scores_pruning_BOOST1")
    for i in range(len(scores_pruning_BOOST1)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_pruning_BOOST1[i])
    BOOST_file.write("\n")
    BOOST_file.write("scores_pruning_BOOST2")
    for i in range(len(scores_pruning_BOOST2)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_pruning_BOOST2[i])
    BOOST_file.write("\n")

    # comprison of no. of boosting estimators
    temp = 100
    scores_boosting_BOOST1 = np.zeros(temp)
    scores_boosting_BOOST2 = np.zeros(temp)
    temp_n_ = np.zeros(temp)
    for i in range(0, temp):
        temp_n_[i] = i+1
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[np.argmax(scores_pruning_BOOST1)])
        BOOST1 = AdaBoostClassifier(base_estimator=DT1, n_estimators=i+1, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        scores_boosting_BOOST1[i] = cross_val_score(BOOST1, data1_X_train, data1_Y_train, cv = 10, scoring = 'accuracy').mean()
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[np.argmax(scores_pruning_BOOST2)])
        BOOST2 = AdaBoostClassifier(base_estimator=DT2, n_estimators=i+1, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        scores_boosting_BOOST2[i] = cross_val_score(BOOST2, data2_X_train, data2_Y_train, cv = 10, scoring = 'accuracy').mean()
        print "Boosting (DT) (boosting analysis):", (i+1)*100.0/temp , "% done."

    fig_BOOST_4 = plt.figure()
    graph_BOOST_4 = fig_BOOST_4.add_subplot(111, )
    graph_BOOST_4.plot(temp_n_, scores_boosting_BOOST1, label='Accuracy (Dataset 1)')
    graph_BOOST_4.plot(temp_n_, scores_boosting_BOOST2, label='Accuracy (Dataset 2)')
    graph_BOOST_4.set_xlabel("Min Leaf Size (% of no. of sample)")
    graph_BOOST_4.set_ylabel("Accuracy")
    plt.title("Accuracy (number of estimators) - Boosting (DT)")
    plt.legend(loc='upper right');
    plt.savefig('graph_BOOST_4.png')

    BOOST_file.write("scores_boosting_BOOST1")
    for i in range(len(scores_boosting_BOOST1)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_boosting_BOOST1[i])
    BOOST_file.write("\n")
    BOOST_file.write("scores_boosting_BOOST2")
    for i in range(len(scores_boosting_BOOST2)):
        BOOST_file.write(";")
        BOOST_file.write("%1.9f" % scores_boosting_BOOST2[i])
    BOOST_file.write("\n")

    # testing set (for comparison)
    start_time_1 = time.time()
    DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[np.argmax(scores_pruning_BOOST1)])
    BOOST1 = AdaBoostClassifier(base_estimator=DT1, n_estimators=int(temp_n_[np.argmax(scores_boosting_BOOST1)]), learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    scores_train_BOOST1 = sum(BOOST1.fit(data1_X_train, data1_Y_train).predict(data1_X_train) == data1_Y_train) * 1.0 / train_n1
    scores_test_BOOST1 = sum(BOOST1.fit(data1_X_train, data1_Y_train).predict(data1_X_test) == data1_Y_test) * 1.0 / test_n1
    elasped_time_1 = time.time() - start_time_1
    start_time_2 = time.time()
    DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[np.argmax(scores_pruning_BOOST2)])
    BOOST2 = AdaBoostClassifier(base_estimator=DT2, n_estimators=int(temp_n_[np.argmax(scores_boosting_BOOST2)]), learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    scores_train_BOOST2 = sum(BOOST1.fit(data2_X_train, data2_Y_train).predict(data2_X_train) == data2_Y_train) * 1.0 / train_n2
    scores_test_BOOST2 = sum(BOOST1.fit(data2_X_train, data2_Y_train).predict(data2_X_test) == data2_Y_test) * 1.0 / test_n2
    elasped_time_2 = time.time() - start_time_2

    BOOST_file.write("n1" + ";")
    BOOST_file.write("%i" % n1)
    BOOST_file.write("\n")
    BOOST_file.write("n2" + ";")
    BOOST_file.write("%i" % n2)
    BOOST_file.write("\n")
    BOOST_file.write("train_n1" + ";")
    BOOST_file.write("%i" % train_n1)
    BOOST_file.write("\n")
    BOOST_file.write("train_n2" + ";")
    BOOST_file.write("%i" % train_n2)
    BOOST_file.write("\n")
    BOOST_file.write("optimal_leaf_size_1" + ";")
    BOOST_file.write("%1.9f" % temp_n[np.argmax(scores_pruning_BOOST1)])
    BOOST_file.write("\n")
    BOOST_file.write("optimal_leaf_size_2" + ";")
    BOOST_file.write("%1.9f" % temp_n[np.argmax(scores_pruning_BOOST2)])
    BOOST_file.write("\n")
    BOOST_file.write("optimal_boosting_size_1" + ";")
    BOOST_file.write("%1.9f" % temp_n_[np.argmax(scores_boosting_BOOST1)])
    BOOST_file.write("\n")
    BOOST_file.write("optimal_boosting_size_2" + ";")
    BOOST_file.write("%1.9f" % temp_n_[np.argmax(scores_boosting_BOOST2)])
    BOOST_file.write("\n")
    BOOST_file.write("scores_train_BOOST1" + ";")
    BOOST_file.write("%1.9f" % scores_train_BOOST1)
    BOOST_file.write("\n")
    BOOST_file.write("scores_test_BOOST1" + ";")
    BOOST_file.write("%1.9f" % scores_test_BOOST1)
    BOOST_file.write("\n")
    BOOST_file.write("scores_train_BOOST2" + ";")
    BOOST_file.write("%1.9f" % scores_train_BOOST2)
    BOOST_file.write("\n")
    BOOST_file.write("scores_test_BOOST2" + ";")
    BOOST_file.write("%1.9f" % scores_test_BOOST2)
    BOOST_file.write("\n")
    BOOST_file.write("elasped_time_1" + ";")
    BOOST_file.write("%1.9f" % elasped_time_1)
    BOOST_file.write("\n")
    BOOST_file.write("elasped_time_2" + ";")
    BOOST_file.write("%1.9f" % elasped_time_2)
    BOOST_file.close()

print "========== END =========="

