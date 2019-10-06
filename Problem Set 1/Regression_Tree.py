import numpy as np
import scipy as sp
from sklearn import tree
data = np.loadtxt(open("housing_data.csv", "rb"), delimiter=";", skiprows=1)
data_X = data[: , 0:-1]
data_Y = data[: , -1]
DT1 = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf = 0.005)
DT1 = DT1.fit(data_X, data_Y)
Y_head = DT1.predict(data_X)
SS_total = sum((data_Y - np.mean(data_Y)) ** 2)
SS_res = sum((data_Y - Y_head) ** 2)
R_sq = 1 - SS_res / SS_total
print R_sq

