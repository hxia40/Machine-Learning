import numpy as np
import time
import mlrose
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 3)
# print(X_train)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print(X_train_scaled)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

# Initialize neural network object and fit object
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'random_hill_climb', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts =100,
                                 random_state = 3)
nn_model1.fit(X_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print('FHC', y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print('FHC',y_test_accuracy)

'''We can potentially improve on the accuracy of our model by tuning the parameters we set when initializing the neural
network object. Suppose we decide to change the optimization algorithm to gradient descent, but leave all other model
parameters unchanged'''

# Initialize neural network object and fit object
nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'gradient_descent', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100,
                                 random_state = 3)
nn_model2.fit(X_train_scaled, y_train_hot)
# Predict labels for train set and assess accuracy
y_train_pred = nn_model2.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print('gradient_descent', y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model2.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print('gradient_descent', y_test_accuracy)


lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [], activation = 'sigmoid',
                                    algorithm = 'random_hill_climb', max_iters = 1000,
                                    bias = True, is_classifier = True, learning_rate= 0.0001,
                                    early_stopping = True, clip_max = 5, max_attempts= 100,
                                    random_state = 3)

# Initialize logistic regression object and fit object
lr_model1 = mlrose.LogisticRegression(algorithm = 'random_hill_climb', max_iters =1000,
                                      bias = True, learning_rate = 0.0001,
                                      early_stopping = True, clip_max = 5, max_attempts = 100,
                                      random_state = 3)
lr_model1.fit(X_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = lr_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = lr_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(y_test_accuracy)

'''This model achieves 19.2% training accuracy and 6.7% test accuracy, which is worse than if we predicted the labels
by selecting values at random.
Nevertheless, as in the previous section, we can potentially improve model accuracy by tuning the parameters set at
initialization.
Suppose we increase our learning rate to 0.01.
'''

# Initialize logistic regression object and fit object
lr_model2 = mlrose.LogisticRegression(algorithm = 'random_hill_climb', max_iters =1000,
                                      bias = True, learning_rate = 0.01,
                                      early_stopping = True, clip_max = 5, max_attempts = 100,
                                      random_state = 3)
lr_model2.fit(X_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = lr_model2.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = lr_model2.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(y_test_accuracy)


'''This results in signficant improvements to both training and test accuracy, 
with training accuracy levels now reaching
68.3% and test accuracy levels reaching 70%.'''
