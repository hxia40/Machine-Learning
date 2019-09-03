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

############################## Neural Networks ##############################
    
    from sklearn.neural_network import MLPClassifier
    ANN_file = open('number_ANN.txt','w')

    # learning curve
    temp = 50
    scores_train_learning_ANN1 = np.zeros(temp)
    scores_test_learning_ANN1 = np.zeros(temp)
    scores_train_learning_ANN2 = np.zeros(temp)
    scores_test_learning_ANN2 = np.zeros(temp)
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
        ANN1 = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, max_iter=500)
        scores_train_learning_ANN1[i] = sum(ANN1.fit(data1_X_temp, data1_Y_temp).predict(data1_X_temp) == data1_Y_temp) * 1.0 / temp_n1_
        scores_test_learning_ANN1[i] = cross_val_score(ANN1, data1_X_temp, data1_Y_temp, cv = 10, scoring = 'accuracy').mean()
        ANN2 = MLPClassifier(hidden_layer_sizes=(5, ), random_state=1, max_iter=500)
        scores_train_learning_ANN2[i] = sum(ANN2.fit(data2_X_temp, data2_Y_temp).predict(data2_X_temp) == data2_Y_temp) * 1.0 / temp_n2_
        scores_test_learning_ANN2[i] = cross_val_score(ANN2, data2_X_temp, data2_Y_temp, cv = 10, scoring = 'accuracy').mean()
        print "Artifical Neural Networks (learning curve):", (i+1)*100.0/temp , "% done."

    fig_ANN_1 = plt.figure()
    graph_ANN_1 = fig_ANN_1.add_subplot(111, )
    graph_ANN_1.plot(temp_n1, scores_train_learning_ANN1, label='Training Set Accuracy')
    graph_ANN_1.plot(temp_n1, scores_test_learning_ANN1, label='Testing Set Accuracy')
    graph_ANN_1.set_xlabel("Sample Size")
    graph_ANN_1.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Artifical Neural Networks (Dataset 1)")
    plt.legend(loc='upper right');
    plt.savefig('graph_ANN_1.png')

    ANN_file.write("scores_train_learning_ANN1")
    for i in range(len(scores_train_learning_ANN1)):
        ANN_file.write(";")
        ANN_file.write("%1.9f" % scores_train_learning_ANN1[i])
    ANN_file.write("\n")
    ANN_file.write("scores_test_learning_ANN1")
    for i in range(len(scores_test_learning_ANN1)):
        ANN_file.write(";")
        ANN_file.write("%1.9f" % scores_test_learning_ANN1[i])
    ANN_file.write("\n")

    fig_ANN_2 = plt.figure()
    graph_ANN_2 = fig_ANN_2.add_subplot(111, )
    graph_ANN_2.plot(temp_n2, scores_train_learning_ANN2, label='Training Set Accuracy')
    graph_ANN_2.plot(temp_n2, scores_test_learning_ANN2, label='Testing Set Accuracy')
    graph_ANN_2.set_xlabel("Sample Size")
    graph_ANN_2.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Artifical Neural Networks (Dataset 2)")
    plt.legend(loc='upper right');
    plt.savefig('graph_ANN_2.png')

    ANN_file.write("scores_train_learning_ANN2")
    for i in range(len(scores_train_learning_ANN2)):
        ANN_file.write(";")
        ANN_file.write("%1.9f" % scores_train_learning_ANN2[i])
    ANN_file.write("\n")
    ANN_file.write("scores_test_learning_ANN2")
    for i in range(len(scores_test_learning_ANN2)):
        ANN_file.write(";")
        ANN_file.write("%1.9f" % scores_test_learning_ANN2[i])
    ANN_file.write("\n")

    # decision of nodes (No. of perceptron analysis)
    temp = 10
    scores_selection_ANN1 = np.zeros(temp)
    scores_selection_ANN2 = np.zeros(temp)
    temp_n = np.zeros(temp)
    for i in range(0, temp):
        temp_n[i] = i+1
        ANN1 = MLPClassifier(hidden_layer_sizes=(i+1, ), random_state=1, max_iter=500)
        scores_selection_ANN1[i] = cross_val_score(ANN1, data1_X_train, data1_Y_train, cv = 10, scoring = 'accuracy').mean()
        ANN2 = MLPClassifier(hidden_layer_sizes=(i+1, ), random_state=1, max_iter=500)
        scores_selection_ANN2[i] = cross_val_score(ANN2, data2_X_train, data2_Y_train, cv = 10, scoring = 'accuracy').mean()
        print "Artifical Neural Networks (No. of Nodes):", (i+1)*100.0/temp , "% done."

    fig_ANN_3 = plt.figure()
    graph_ANN_3 = fig_ANN_3.add_subplot(111, )
    graph_ANN_3.plot(temp_n, scores_selection_ANN1, label='Accuracy (Dataset 1)')
    graph_ANN_3.plot(temp_n, scores_selection_ANN2, label='Accuracy (Dataset 2)')
    graph_ANN_3.set_xlabel("No. of Nodes")
    graph_ANN_3.set_ylabel("Accuracy")
    plt.title("Accuracy (different no of nodes) - Artifical Neural Networks")
    plt.legend(loc='upper right');
    plt.savefig('graph_ANN_3.png')

    ANN_file.write("scores_selection_ANN1")
    for i in range(len(scores_selection_ANN1)):
        ANN_file.write(";")
        ANN_file.write("%1.9f" % scores_selection_ANN1[i])
    ANN_file.write("\n")
    ANN_file.write("scores_selection_ANN2")
    for i in range(len(scores_selection_ANN2)):
        ANN_file.write(";")
        ANN_file.write("%1.9f" % scores_selection_ANN2[i])
    ANN_file.write("\n")

    # testing set (for comparison)
    start_time_1 = time.time()
    ANN1 = MLPClassifier(hidden_layer_sizes=(int(temp_n[np.argmax(scores_selection_ANN1)]), ), random_state=1, max_iter=500)
    scores_train_ANN1 = sum(ANN1.fit(data1_X_train, data1_Y_train).predict(data1_X_train) == data1_Y_train) * 1.0 / train_n1
    scores_test_ANN1 = sum(ANN1.fit(data1_X_train, data1_Y_train).predict(data1_X_test) == data1_Y_test) * 1.0 / test_n1
    elasped_time_1 = time.time() - start_time_1
    start_time_2 = time.time()
    ANN2 = MLPClassifier(hidden_layer_sizes=(int(temp_n[np.argmax(scores_selection_ANN2)]), ), random_state=1, max_iter=500)
    scores_train_ANN2 = sum(ANN1.fit(data2_X_train, data2_Y_train).predict(data2_X_train) == data2_Y_train) * 1.0 / train_n2
    scores_test_ANN2 = sum(ANN1.fit(data2_X_train, data2_Y_train).predict(data2_X_test) == data2_Y_test) * 1.0 / test_n2
    elasped_time_2 = time.time() - start_time_2

    ANN_file.write("n1" + ";")
    ANN_file.write("%i" % n1)
    ANN_file.write("\n")
    ANN_file.write("n2" + ";")
    ANN_file.write("%i" % n2)
    ANN_file.write("\n")
    ANN_file.write("train_n1" + ";")
    ANN_file.write("%i" % train_n1)
    ANN_file.write("\n")
    ANN_file.write("train_n2" + ";")
    ANN_file.write("%i" % train_n2)
    ANN_file.write("\n")
    ANN_file.write("optimal_nodes_1" + ";")
    ANN_file.write("%1.9f" % int(temp_n[np.argmax(scores_selection_ANN1)]))
    ANN_file.write("\n")
    ANN_file.write("optimal_nodes_2" + ";")
    ANN_file.write("%1.9f" % int(temp_n[np.argmax(scores_selection_ANN2)]))
    ANN_file.write("\n")
    ANN_file.write("scores_train_ANN1" + ";")
    ANN_file.write("%1.9f" % scores_train_ANN1)
    ANN_file.write("\n")
    ANN_file.write("scores_test_ANN1" + ";")
    ANN_file.write("%1.9f" % scores_test_ANN1)
    ANN_file.write("\n")
    ANN_file.write("scores_train_ANN2" + ";")
    ANN_file.write("%1.9f" % scores_train_ANN2)
    ANN_file.write("\n")
    ANN_file.write("scores_test_ANN2" + ";")
    ANN_file.write("%1.9f" % scores_test_ANN2)
    ANN_file.write("\n")
    ANN_file.write("elasped_time_1" + ";")
    ANN_file.write("%1.9f" % elasped_time_1)
    ANN_file.write("\n")
    ANN_file.write("elasped_time_2" + ";")
    ANN_file.write("%1.9f" % elasped_time_2)
    ANN_file.close()

print "========== END =========="

