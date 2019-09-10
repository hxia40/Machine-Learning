import numpy as np
import pandas as pd
import datetime as SVM
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

############################## Support Vector Machines ##############################

    from sklearn import svm
    SVM_file = open('number_SVM.txt','w')

    # learning curve
    temp = 50
    scores_train_learning_SVM1 = np.zeros(temp)
    scores_test_learning_SVM1 = np.zeros(temp)
    scores_train_learning_SVM2 = np.zeros(temp)
    scores_test_learning_SVM2 = np.zeros(temp)
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
        SVM1 = svm.SVC(kernel='rbf')
        scores_train_learning_SVM1[i] = sum(SVM1.fit(data1_X_temp, data1_Y_temp).predict(data1_X_temp) == data1_Y_temp) * 1.0 / temp_n1_
        scores_test_learning_SVM1[i] = cross_val_score(SVM1, data1_X_temp, data1_Y_temp, cv = 10, scoring = 'accuracy').mean()
        SVM2 = svm.SVC(kernel='rbf')
        scores_train_learning_SVM2[i] = sum(SVM2.fit(data2_X_temp, data2_Y_temp).predict(data2_X_temp) == data2_Y_temp) * 1.0 / temp_n2_
        scores_test_learning_SVM2[i] = cross_val_score(SVM2, data2_X_temp, data2_Y_temp, cv = 10, scoring = 'accuracy').mean()
        print "Support Vector Machines (learning curve):", (i+1)*100.0/temp , "% done."

    fig_SVM_1 = plt.figure()
    graph_SVM_1 = fig_SVM_1.add_subplot(111, )
    graph_SVM_1.plot(temp_n1, scores_train_learning_SVM1, label='Training Set Accuracy')
    graph_SVM_1.plot(temp_n1, scores_test_learning_SVM1, label='Testing Set Accuracy')
    graph_SVM_1.set_xlabel("Sample Size")
    graph_SVM_1.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Support Vector Machines (Dataset 1)")
    plt.legend(loc='upper right');
    plt.savefig('graph_SVM_1.png')

    SVM_file.write("scores_train_learning_SVM1")
    for i in range(len(scores_train_learning_SVM1)):
        SVM_file.write(";")
        SVM_file.write("%1.9f" % scores_train_learning_SVM1[i])
    SVM_file.write("\n")
    SVM_file.write("scores_test_learning_SVM1")
    for i in range(len(scores_test_learning_SVM1)):
        SVM_file.write(";")
        SVM_file.write("%1.9f" % scores_test_learning_SVM1[i])
    SVM_file.write("\n")

    fig_SVM_2 = plt.figure()
    graph_SVM_2 = fig_SVM_2.add_subplot(111, )
    graph_SVM_2.plot(temp_n2, scores_train_learning_SVM2, label='Training Set Accuracy')
    graph_SVM_2.plot(temp_n2, scores_test_learning_SVM2, label='Testing Set Accuracy')
    graph_SVM_2.set_xlabel("Sample Size")
    graph_SVM_2.set_ylabel("Accuracy")
    plt.title("Accuracy (Training vs Testing) - Support Vector Machines (Dataset 2)")
    plt.legend(loc='upper right');
    plt.savefig('graph_SVM_2.png')

    SVM_file.write("scores_train_learning_SVM2")
    for i in range(len(scores_train_learning_SVM2)):
        SVM_file.write(";")
        SVM_file.write("%1.9f" % scores_train_learning_SVM2[i])
    SVM_file.write("\n")
    SVM_file.write("scores_test_learning_SVM2")
    for i in range(len(scores_test_learning_SVM2)):
        SVM_file.write(";")
        SVM_file.write("%1.9f" % scores_test_learning_SVM2[i])
    SVM_file.write("\n")

    # decision of kernel (selection)
    kernel_list = ['rbf', 'sigmoid']
    temp = len(kernel_list)
    scores_kernel_SVM1 = np.zeros(temp)
    scores_kernel_SVM2 = np.zeros(temp)
    for i in range(0, temp):
        SVM1 = svm.SVC(kernel=kernel_list[i])
        scores_kernel_SVM1[i] = cross_val_score(SVM1, data1_X_train, data1_Y_train, cv = 10, scoring = 'accuracy').mean()
        SVM2 = svm.SVC(kernel=kernel_list[i])
        scores_kernel_SVM2[i] = cross_val_score(SVM2, data2_X_train, data2_Y_train, cv = 10, scoring = 'accuracy').mean()
        print "Support Vector Machines (kernel selection):", (i+1)*100.0/temp , "% done."

    fig_SVM_3 = plt.figure()
    graph_SVM_3 = fig_SVM_3.add_subplot(111, )
    temp = list(range(0, temp))
    plt.xticks(temp, kernel_list)
    graph_SVM_3.plot(temp, scores_kernel_SVM1, label='Accuracy (Dataset 1)')
    graph_SVM_3.plot(temp, scores_kernel_SVM2, label='Accuracy (Dataset 2)')
    graph_SVM_3.set_xlabel("Kernel")
    graph_SVM_3.set_ylabel("Accuracy")
    plt.title("Accuracy (different kernel) - Support Vector Machines")
    plt.legend(loc='upper right');
    plt.savefig('graph_SVM_3.png')

    SVM_file.write("scores_kernel_SVM1")
    for i in range(len(scores_kernel_SVM1)):
        SVM_file.write(";")
        SVM_file.write("%1.9f" % scores_kernel_SVM1[i])
    SVM_file.write("\n")
    SVM_file.write("scores_kernel_SVM2")
    for i in range(len(scores_kernel_SVM2)):
        SVM_file.write(";")
        SVM_file.write("%1.9f" % scores_kernel_SVM2[i])
    SVM_file.write("\n")

    # testing set (for comparison)
    start_time_1 = time.time()
    SVM1 = svm.SVC(kernel=kernel_list[np.argmax(scores_kernel_SVM1)])
    scores_train_SVM1 = sum(SVM1.fit(data1_X_train, data1_Y_train).predict(data1_X_train) == data1_Y_train) * 1.0 / train_n1
    scores_test_SVM1 = sum(SVM1.fit(data1_X_train, data1_Y_train).predict(data1_X_test) == data1_Y_test) * 1.0 / test_n1
    elasped_time_1 = time.time() - start_time_1
    start_time_2 = time.time()
    SVM2 = svm.SVC(kernel=kernel_list[np.argmax(scores_kernel_SVM2)])
    scores_train_SVM2 = sum(SVM1.fit(data2_X_train, data2_Y_train).predict(data2_X_train) == data2_Y_train) * 1.0 / train_n2
    scores_test_SVM2 = sum(SVM1.fit(data2_X_train, data2_Y_train).predict(data2_X_test) == data2_Y_test) * 1.0 / test_n2
    elasped_time_2 = time.time() - start_time_2

    SVM_file.write("n1" + ";")
    SVM_file.write("%i" % n1)
    SVM_file.write("\n")
    SVM_file.write("n2" + ";")
    SVM_file.write("%i" % n2)
    SVM_file.write("\n")
    SVM_file.write("train_n1" + ";")
    SVM_file.write("%i" % train_n1)
    SVM_file.write("\n")
    SVM_file.write("train_n2" + ";")
    SVM_file.write("%i" % train_n2)
    SVM_file.write("\n")
    SVM_file.write("optimal_kernel_1" + ";")
    SVM_file.write(kernel_list[np.argmax(scores_kernel_SVM1)])
    SVM_file.write("\n")
    SVM_file.write("optimal_kernel_2" + ";")
    SVM_file.write(kernel_list[np.argmax(scores_kernel_SVM2)])
    SVM_file.write("\n")
    SVM_file.write("scores_train_SVM1" + ";")
    SVM_file.write("%1.9f" % scores_train_SVM1)
    SVM_file.write("\n")
    SVM_file.write("scores_test_SVM1" + ";")
    SVM_file.write("%1.9f" % scores_test_SVM1)
    SVM_file.write("\n")
    SVM_file.write("scores_train_SVM2" + ";")
    SVM_file.write("%1.9f" % scores_train_SVM2)
    SVM_file.write("\n")
    SVM_file.write("scores_test_SVM2" + ";")
    SVM_file.write("%1.9f" % scores_test_SVM2)
    SVM_file.write("\n")
    SVM_file.write("elasped_time_1" + ";")
    SVM_file.write("%1.9f" % elasped_time_1)
    SVM_file.write("\n")
    SVM_file.write("elasped_time_2" + ";")
    SVM_file.write("%1.9f" % elasped_time_2)
    SVM_file.close()

print "========== END =========="

