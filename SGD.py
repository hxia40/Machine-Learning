from sklearn.datasets import load_iris
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# load data - iris flower data
iris = load_iris()
print "iris\n", iris

# choose features and tags
X_iris, y_iris = iris.data, iris.target

print "\nX_iris:\n", X_iris, "\ny_iris:\n", y_iris
# choose the first two columns as features
X, y = X_iris[:, :2], y_iris

# choose 25% of the data as training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 33)

# standardize the original data - this is important but usually neglected by newbies.
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import SGDClassifier

# choose SGDClassifier, which suits large-scale data, which use Stochastic Gradient Descent method to estimate parameter
clf = SGDClassifier()

clf.fit(X_train, y_train)

# import metrics, which evaluate how good the performance is
from sklearn import metrics

y_train_predict = clf.predict(X_train)

# Internal test, using training set for accurate performance evaluation
print metrics.accuracy_score(y_train, y_train_predict)

# Standard external test, using test set for accurate performance evaluation
y_predict = clf.predict(X_test)
print metrics.accuracy_score(y_test, y_predict)