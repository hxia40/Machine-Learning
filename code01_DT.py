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

############################## Decision Tree ##############################

    from sklearn import tree
    DT_file = open('number_DT.txt','w')

    # learning curve
    temp = 100
    scores_train_learning_DT1 = np.zeros(temp)
    scores_test_learning_DT1 = np.zeros(temp)
    scores_train_learning_DT2 = np.zeros(temp)
    scores_test_learning_DT2 = np.zeros(temp)
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
        scores_train_learning_DT1[i] = sum(DT1.fit(data1_X_temp, data1_Y_temp).predict(data1_X_temp) == data1_Y_temp) * 1.0 / temp_n1_
        scores_test_learning_DT1[i] = cross_val_score(DT1, data1_X_temp, data1_Y_temp, cv = 10, scoring = 'accuracy').mean()
        DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = 0.005)
        scores_train_learning_DT2[i] = sum(DT2.fit(data2_X_temp, data2_Y_temp).predict(data2_X_temp) == data2_Y_temp) * 1.0 / temp_n2_
        scores_test_learning_DT2[i] = cross_val_score(DT2, data2_X_temp, data2_Y_temp, cv = 10, scoring = 'accuracy').mean()
        print "Decision Tree (learning curve):", (i+1)*100.0/temp , "% done."

    fig_DT_1 = plt.figure()
    graph_DT_1 = fig_DT_1.add_subplot(111, )
    graph_DT_1.plot(temp_n1, scores_train_learning_DT1, label='Training Set Accuracy')
    graph_DT_1.plot(temp_n1, scores_test_learning_DT1, label='Testing Set Accuracy')
    graph_DT_1.set_xlabel("Sample Size")
    graph_DT_1.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Decision Tree (Dataset 1)")
    plt.legend(loc='upper right');
    plt.savefig('graph_DT_1.png')

    DT_file.write("scores_train_learning_DT1")
    for i in range(len(scores_train_learning_DT1)):
        DT_file.write(";")
        DT_file.write("%1.9f" % scores_train_learning_DT1[i])
    DT_file.write("\n")
    DT_file.write("scores_test_learning_DT1")
    for i in range(len(scores_test_learning_DT1)):
        DT_file.write(";")
        DT_file.write("%1.9f" % scores_test_learning_DT1[i])
    DT_file.write("\n")

    fig_DT_2 = plt.figure()
    graph_DT_2 = fig_DT_2.add_subplot(111, )
    graph_DT_2.plot(temp_n2, scores_train_learning_DT2, label='Training Set Accuracy')
    graph_DT_2.plot(temp_n2, scores_test_learning_DT2, label='Testing Set Accuracy')
    graph_DT_2.set_xlabel("Sample Size")
    graph_DT_2.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Decision Tree (Dataset 2)")
    plt.legend(loc='upper right');
    plt.savefig('graph_DT_2.png')

    DT_file.write("scores_train_learning_DT2")
    for i in range(len(scores_train_learning_DT2)):
        DT_file.write(";")
        DT_file.write("%1.9f" % scores_train_learning_DT2[i])
    DT_file.write("\n")
    DT_file.write("scores_test_learning_DT2")
    for i in range(len(scores_test_learning_DT2)):
        DT_file.write(";")
        DT_file.write("%1.9f" % scores_test_learning_DT2[i])
    DT_file.write("\n")

    # decision of leaf size (pruning)
    temp = 100
    scores_pruning_DT1 = np.zeros(temp)
    scores_pruning_DT2 = np.zeros(temp)
    temp_n = np.zeros(temp)
    for i in range(0, temp):
        temp_n[i] = (i+1) * 0.25 / temp
        DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[i])
        scores_pruning_DT1[i] = cross_val_score(DT1, data1_X_train, data1_Y_train, cv = 10, scoring = 'accuracy').mean()
        DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[i])
        scores_pruning_DT2[i] = cross_val_score(DT2, data2_X_train, data2_Y_train, cv = 10, scoring = 'accuracy').mean()
        print "Decision Tree (pruning analysis):", (i+1)*100.0/temp , "% done."

    fig_DT_3 = plt.figure()
    graph_DT_3 = fig_DT_3.add_subplot(111, )
    graph_DT_3.plot(temp_n, scores_pruning_DT1, label='Accuracy (Dataset 1)')
    graph_DT_3.plot(temp_n, scores_pruning_DT2, label='Accuracy (Dataset 2)')
    graph_DT_3.set_xlabel("Min Leaf Size (% of no. of sample)")
    graph_DT_3.set_ylabel("Accuracy")
    plt.title("Accuracy (different minimum leaf size) - Decision Tree")
    plt.legend(loc='upper right');
    plt.savefig('graph_DT_3.png')

    DT_file.write("scores_pruning_DT1")
    for i in range(len(scores_pruning_DT1)):
        DT_file.write(";")
        DT_file.write("%1.9f" % scores_pruning_DT1[i])
    DT_file.write("\n")
    DT_file.write("scores_pruning_DT2")
    for i in range(len(scores_pruning_DT2)):
        DT_file.write(";")
        DT_file.write("%1.9f" % scores_pruning_DT2[i])
    DT_file.write("\n")

    # testing set (for comparison)
    start_time_1 = time.time()
    DT1 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[np.argmax(scores_pruning_DT1)])
    scores_train_DT1 = sum(DT1.fit(data1_X_train, data1_Y_train).predict(data1_X_train) == data1_Y_train) * 1.0 / train_n1
    scores_test_DT1 = sum(DT1.fit(data1_X_train, data1_Y_train).predict(data1_X_test) == data1_Y_test) * 1.0 / test_n1
    elasped_time_1 = time.time() - start_time_1
    start_time_2 = time.time()
    DT2 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf = temp_n[np.argmax(scores_pruning_DT2)])
    scores_train_DT2 = sum(DT1.fit(data2_X_train, data2_Y_train).predict(data2_X_train) == data2_Y_train) * 1.0 / train_n2
    scores_test_DT2 = sum(DT1.fit(data2_X_train, data2_Y_train).predict(data2_X_test) == data2_Y_test) * 1.0 / test_n2
    elasped_time_2 = time.time() - start_time_2

    DT_file.write("n1" + ";")
    DT_file.write("%i" % n1)
    DT_file.write("\n")
    DT_file.write("n2" + ";")
    DT_file.write("%i" % n2)
    DT_file.write("\n")
    DT_file.write("train_n1" + ";")
    DT_file.write("%i" % train_n1)
    DT_file.write("\n")
    DT_file.write("train_n2" + ";")
    DT_file.write("%i" % train_n2)
    DT_file.write("\n")
    DT_file.write("optimal_leaf_size_1" + ";")
    DT_file.write("%1.9f" % temp_n[np.argmax(scores_pruning_DT1)])
    DT_file.write("\n")
    DT_file.write("optimal_leaf_size_2" + ";")
    DT_file.write("%1.9f" % temp_n[np.argmax(scores_pruning_DT2)])
    DT_file.write("\n")
    DT_file.write("scores_train_DT1" + ";")
    DT_file.write("%1.9f" % scores_train_DT1)
    DT_file.write("\n")
    DT_file.write("scores_test_DT1" + ";")
    DT_file.write("%1.9f" % scores_test_DT1)
    DT_file.write("\n")
    DT_file.write("scores_train_DT2" + ";")
    DT_file.write("%1.9f" % scores_train_DT2)
    DT_file.write("\n")
    DT_file.write("scores_test_DT2" + ";")
    DT_file.write("%1.9f" % scores_test_DT2)
    DT_file.write("\n")
    DT_file.write("elasped_time_1" + ";")
    DT_file.write("%1.9f" % elasped_time_1)
    DT_file.write("\n")
    DT_file.write("elasped_time_2" + ";")
    DT_file.write("%1.9f" % elasped_time_2)
    DT_file.close()

print "========== END =========="

