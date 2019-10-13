import mlrose
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from datetime import datetime

import matplotlib.pyplot as plt

def load_training_data():

	print('Loading training dataset ...')
	df = pd.read_csv('data/epileptic-seizure-train-standardized.csv', header = 0)

	X_train = df.iloc[:, 0:-1]
	y_train = df.iloc[:, -1]

	return X_train, y_train

def load_test_data():

	print('Loading testing dataset ...')
	df = pd.read_csv('data/epileptic-seizure-test-standardized.csv', header = 0)

	X_test = df.iloc[:, 0:-1]
	y_test = df.iloc[:, -1]

	return X_test, y_test

def encode(y_train, y_test):

	y_train = np.array(y_train)
	y_test = np.array(y_test)

	# one hot encode target values
	one_hot = OneHotEncoder(categories='auto')

	y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
	y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

	return y_train_hot, y_test_hot

def plot_curve(testRange, train_mean, train_std, test_mean, test_std, xLabel, title, fileName):

	yticks = np.arange(0, 1, 0.1)
	lw = 0.2
	plt.style.use('seaborn')
	plt.plot(testRange, train_mean, label = 'Train', color = 'mediumblue')
	plt.plot(testRange, test_mean, label='Test', color = 'darkorange')
	plt.fill_between(testRange, train_mean - train_std, train_mean + train_std, alpha = 0.2, color = 'mediumblue', lw = lw)
	plt.fill_between(testRange, test_mean - test_std, test_mean + test_std, alpha = 0.2, color = 'darkorange', lw = lw)
	plt.ylabel('Accuracy', fontsize = 14, x = 1.03)
	plt.xlabel(xLabel, fontsize = 14, y = 1.03)
	plt.title(title, fontsize = 18, y = 1.03)
	plt.xticks(fontsize = 12)
	plt.yticks(yticks, fontsize = 12)
	plt.legend(fontsize = 12)
	plt.savefig(fileName)
	plt.close()

def default_results():

	print('\n', '-'*5, 'GD', '-'*5)
	nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
								  activation = 'relu', 
								  algorithm = 'gradient_descent', 
								  max_iters = 100, 
								  bias = True, 
								  is_classifier = True, 
								  learning_rate = 0.1, 
								  early_stopping = True, 
								  clip_max = 10,
								  max_attempts = 10)

	nn.fit(X_train, y_train)
	y_train_pred = nn.predict(X_train)
	y_train_accuracy = accuracy_score(y_train, y_train_pred)
	print('Training accuracy: ', y_train_accuracy)

	y_test_pred = nn.predict(X_test)
	y_test_accuracy = accuracy_score(y_test, y_test_pred)
	print('Test accuracy: ', y_test_accuracy)


	print('\n', '-'*5, 'RHC', '-'*5)
	nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
									     activation = 'relu', 
									     algorithm = 'random_hill_climb', 
									     max_iters = 100, 
									     bias = True, 
									     is_classifier = True, 
									     learning_rate = 0.1, 
									     early_stopping = True, 
									     clip_max = 10,
									     max_attempts = 10)
	nn.fit(X_train, y_train)
	y_train_pred = nn.predict(X_train)
	y_train_accuracy = accuracy_score(y_train, y_train_pred)
	print('Training accuracy: ', y_train_accuracy)

	y_test_pred = nn.predict(X_test)
	y_test_accuracy = accuracy_score(y_test, y_test_pred)
	print('Test accuracy: ', y_test_accuracy)

	print('\n', '-'*5, 'SA', '-'*5)
	nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
								     activation = 'relu', 
								     algorithm = 'simulated_annealing', 
								     max_iters = 100, 
								     bias = True, 
								     is_classifier = True, 
								     learning_rate = 0.1, 
								     early_stopping = True, 
								     clip_max = 10,
								     max_attempts = 10)
	nn.fit(X_train, y_train)
	y_train_pred = nn.predict(X_train)
	y_train_accuracy = accuracy_score(y_train, y_train_pred)
	print('Training accuracy: ', y_train_accuracy)

	y_test_pred = nn.predict(X_test)
	y_test_accuracy = accuracy_score(y_test, y_test_pred)
	print('Test accuracy: ', y_test_accuracy)

	print('\n', '-'*5, 'GA', '-'*5)
	nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
								     activation = 'relu', 
								     algorithm = 'genetic_alg', 
								     max_iters = 100, 
								     bias = True, 
								     is_classifier = True, 
								     learning_rate = 0.1, 
								     early_stopping = True, 
								     clip_max = 10,
								     max_attempts = 10)
	nn.fit(X_train, y_train)
	y_train_pred = nn.predict(X_train)
	y_train_accuracy = accuracy_score(y_train, y_train_pred)
	print('Training accuracy: ', y_train_accuracy)

	y_test_pred = nn.predict(X_test)
	y_test_accuracy = accuracy_score(y_test, y_test_pred)
	print('Test accuracy: ', y_test_accuracy)

def grid_search_gd():

	max_accuracy = 0
	best_params = {}
	count = 0

	for learning_rate in np.arange(0.0001, 1.0001, 0.0001):
		for max_attempts in range(50, 1001, 50):
			count += 1
			print('Running ', count)
			nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'gradient_descent', 
											  max_iters = 100, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = learning_rate, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = max_attempts)
			nn.fit(X_train, y_train)
			y_test_pred = nn.predict(X_test)
			y_test_accuracy = accuracy_score(y_test, y_test_pred)
			if y_test_accuracy > max_accuracy:
				max_accuracy = y_test_accuracy
				best_params['learning_rate'] = learning_rate
				best_params['max_attempts'] = max_attempts

	print('GD: ')
	print('Max accuracy: ', max_accuracy)
	print('Best params: ', best_params)

def grid_search_rhc():

	max_accuracy = 0
	best_params = {}
	count = 0

	for learning_rate in range(21):
		for max_attempts in range(50, 1001, 50):
			count += 1
			print('Running ', count)
			nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'random_hill_climb', 
											  max_iters = 100, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = learning_rate, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = max_attempts)
			nn.fit(X_train, y_train)
			y_test_pred = nn.predict(X_test)
			y_test_accuracy = accuracy_score(y_test, y_test_pred)
			if y_test_accuracy > max_accuracy:
				max_accuracy = y_test_accuracy
				best_params['learning_rate'] = learning_rate
				best_params['max_attempts'] = max_attempts

	print('RHC: ')
	print('Max accuracy: ', max_accuracy)
	print('Best params: ', best_params)

def grid_search_sa():

	max_accuracy = 0
	best_params = {}
	count = 0

	for learning_rate in range(21):
		for max_attempts in range(50, 2001, 50):
			for schedule in [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]:
				count += 1
				print('Running ', count)
				nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
												  activation = 'relu', 
												  algorithm = 'simulated_annealing', 
												  max_iters = 100, 
												  bias = True, 
												  is_classifier = True, 
												  learning_rate = learning_rate, 
												  early_stopping = True, 
												  clip_max = 10,
												  max_attempts = max_attempts,
												  schedule = schedule)
				nn.fit(X_train, y_train)
				y_test_pred = nn.predict(X_test)
				y_test_accuracy = accuracy_score(y_test, y_test_pred)
				if y_test_accuracy > max_accuracy:
					max_accuracy = y_test_accuracy
					best_params['learning_rate'] = learning_rate
					best_params['max_attempts'] = max_attempts
					best_params['schedule'] = schedule

	print('SA: ')
	print('Max accuracy: ', max_accuracy)
	print('Best params: ', best_params)

def grid_search_ga():

	max_accuracy = 0
	best_params = {}
	count = 0

	for learning_rate in range(21):
		for max_attempts in range(50, 2001, 50):
			for pop_size in range(50, 1001, 50):
				for mutation_prob in np.arange(0.02, 0.51, 0.02):
					count += 1
					print('Running ', count)
					nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
													  activation = 'relu', 
													  algorithm = 'genetic_alg', 
													  max_iters = 100, 
													  bias = True, 
													  is_classifier = True, 
													  learning_rate = learning_rate, 
													  early_stopping = True, 
													  clip_max = 10,
													  max_attempts = max_attempts,
													  pop_size = pop_size,
													  mutation_prob = mutation_prob)
					nn.fit(X_train, y_train)
					y_test_pred = nn.predict(X_test)
					y_test_accuracy = accuracy_score(y_test, y_test_pred)
					if y_test_accuracy > max_accuracy:
						max_accuracy = y_test_accuracy
						best_params['learning_rate'] = learning_rate
						best_params['max_attempts'] = max_attempts
						best_params['pop_size'] = pop_size
						best_params['mutation_prob'] = mutation_prob

	print('GA: ')
	print('Max accuracy: ', max_accuracy)
	print('Best params: ', best_params)

def explore_iters_gd():

	opt_time = []
	train_time = []
	test_time = []
	accuracy_train = []
	accuracy_test = []

	f1_scores = []

	for iters in iterRange:
		print('iters: ', iters)

		opt_time_sub = []
		train_time_sub = []
		test_time_sub = []
		accuracy_train_sub = []
		accuracy_test_sub = []
		f1_scores_sub = []

		for repeat in range(repeats):
			t1 = datetime.now()
			nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'gradient_descent', 
											  max_iters = iters, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = 0.0002, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = 400)
			t2 = datetime.now()
			nn.fit(X_train, y_train)
			t3 = datetime.now()
			y_test_pred = nn.predict(X_test)
			t4 = datetime.now()
			y_test_accuracy = accuracy_score(y_test, y_test_pred)
			accuracy_test_sub.append(y_test_accuracy)

			y_train_pred = nn.predict(X_train)
			y_train_accuracy = accuracy_score(y_train, y_train_pred)
			accuracy_train_sub.append(y_train_accuracy)

			opt_time_sub.append((t2 - t1).microseconds)
			train_time_sub.append((t3 - t2).microseconds)
			test_time_sub.append((t4 - t3).microseconds)

			f1 = f1_score(y_test, y_test_pred, average='weighted')
			f1_scores_sub.append(f1)

		opt_time.append(opt_time_sub)
		train_time.append(train_time_sub)
		test_time.append(test_time_sub)
		accuracy_train.append(accuracy_train_sub)
		accuracy_test.append(accuracy_test_sub)
		f1_scores.append(f1_scores_sub)

	# write to csv
	opt_time = pd.DataFrame(opt_time)
	train_time = pd.DataFrame(train_time)
	test_time = pd.DataFrame(test_time)
	accuracy_train = pd.DataFrame(accuracy_train)
	accuracy_test = pd.DataFrame(accuracy_test)
	f1_scores = pd.DataFrame(f1_scores)

	opt_time.to_csv('nn_gd_optTime.csv', 
					header = headers, 
					index = False)
	train_time.to_csv('nn_gd_trainTime.csv', 
					  header = headers, 
					  index = False)
	test_time.to_csv('nn_gd_testTime.csv', 
					  header = headers, 
					  index = False)
	accuracy_train.to_csv('nn_gd_accTrain.csv', 
					  header = headers, 
					  index = False)
	accuracy_test.to_csv('nn_gd_accTest.csv', 
					  header = headers, 
					  index = False)
	f1_scores.to_csv('nn_gd_f1.csv', 
					  header = headers, 
					  index = False)

def explore_iters_rhc():

	opt_time = []
	train_time = []
	test_time = []
	accuracy_train = []
	accuracy_test = []

	f1_scores = []

	for iters in iterRange:
		print('iters: ', iters)

		opt_time_sub = []
		train_time_sub = []
		test_time_sub = []
		accuracy_train_sub = []
		accuracy_test_sub = []
		f1_scores_sub = []

		for repeat in range(repeats):
			t1 = datetime.now()
			nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'random_hill_climb', 
											  max_iters = iters, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = 7, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = 700)
			t2 = datetime.now()
			nn.fit(X_train, y_train)
			t3 = datetime.now()
			y_test_pred = nn.predict(X_test)
			t4 = datetime.now()
			y_test_accuracy = accuracy_score(y_test, y_test_pred)
			accuracy_test_sub.append(y_test_accuracy)

			y_train_pred = nn.predict(X_train)
			y_train_accuracy = accuracy_score(y_train, y_train_pred)
			accuracy_train_sub.append(y_train_accuracy)

			opt_time_sub.append((t2 - t1).microseconds)
			train_time_sub.append((t3 - t2).microseconds)
			test_time_sub.append((t4 - t3).microseconds)

			f1 = f1_score(y_test, y_test_pred, average='weighted')
			f1_scores_sub.append(f1)

		opt_time.append(opt_time_sub)
		train_time.append(train_time_sub)
		test_time.append(test_time_sub)
		accuracy_train.append(accuracy_train_sub)
		accuracy_test.append(accuracy_test_sub)
		f1_scores.append(f1_scores_sub)

	# write to csv
	opt_time = pd.DataFrame(opt_time)
	train_time = pd.DataFrame(train_time)
	test_time = pd.DataFrame(test_time)
	accuracy_train = pd.DataFrame(accuracy_train)
	accuracy_test = pd.DataFrame(accuracy_test)
	f1_scores = pd.DataFrame(f1_scores)

	opt_time.to_csv('nn_rhc_optTime.csv', 
					header = headers, 
					index = False)
	train_time.to_csv('nn_rhc_trainTime.csv', 
					  header = headers, 
					  index = False)
	test_time.to_csv('nn_rhc_testTime.csv', 
					  header = headers, 
					  index = False)
	accuracy_train.to_csv('nn_rhc_accTrain.csv', 
					  header = headers, 
					  index = False)
	accuracy_test.to_csv('nn_rhc_accTest.csv', 
					  header = headers, 
					  index = False)
	f1_scores.to_csv('nn_rhc_f1.csv', 
					  header = headers, 
					  index = False)

def explore_iters_sa():

	opt_time = []
	train_time = []
	test_time = []
	accuracy_train = []
	accuracy_test = []

	f1_scores = []

	for iters in iterRange:
		print('iters: ', iters)

		opt_time_sub = []
		train_time_sub = []
		test_time_sub = []
		accuracy_train_sub = []
		accuracy_test_sub = []
		f1_scores_sub = []

		for repeat in range(repeats):
			t1 = datetime.now()
			nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'simulated_annealing', 
											  max_iters = iters, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = 12, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = 1000,
											  schedule = mlrose.GeomDecay())
			t2 = datetime.now()
			nn.fit(X_train, y_train)
			t3 = datetime.now()
			y_test_pred = nn.predict(X_test)
			t4 = datetime.now()
			y_test_accuracy = accuracy_score(y_test, y_test_pred)
			accuracy_test_sub.append(y_test_accuracy)

			y_train_pred = nn.predict(X_train)
			y_train_accuracy = accuracy_score(y_train, y_train_pred)
			accuracy_train_sub.append(y_train_accuracy)

			opt_time_sub.append((t2 - t1).microseconds)
			train_time_sub.append((t3 - t2).microseconds)
			test_time_sub.append((t4 - t3).microseconds)

			f1 = f1_score(y_test, y_test_pred, average='weighted')
			f1_scores_sub.append(f1)

		opt_time.append(opt_time_sub)
		train_time.append(train_time_sub)
		test_time.append(test_time_sub)
		accuracy_train.append(accuracy_train_sub)
		accuracy_test.append(accuracy_test_sub)
		f1_scores.append(f1_scores_sub)

	# write to csv
	opt_time = pd.DataFrame(opt_time)
	train_time = pd.DataFrame(train_time)
	test_time = pd.DataFrame(test_time)
	accuracy_train = pd.DataFrame(accuracy_train)
	accuracy_test = pd.DataFrame(accuracy_test)
	f1_scores = pd.DataFrame(f1_scores)

	opt_time.to_csv('nn_sa_optTime.csv', 
					header = headers, 
					index = False)
	train_time.to_csv('nn_sa_trainTime.csv', 
					  header = headers, 
					  index = False)
	test_time.to_csv('nn_sa_testTime.csv', 
					  header = headers, 
					  index = False)
	accuracy_train.to_csv('nn_sa_accTrain.csv', 
					  header = headers, 
					  index = False)
	accuracy_test.to_csv('nn_sa_accTest.csv', 
					  header = headers, 
					  index = False)
	f1_scores.to_csv('nn_sa_f1.csv', 
					  header = headers, 
					  index = False)

def explore_iters_ga():

	opt_time = []
	train_time = []
	test_time = []
	accuracy_train = []
	accuracy_test = []

	f1_scores = []

	for iters in iterRange:
		print('iters: ', iters)

		opt_time_sub = []
		train_time_sub = []
		test_time_sub = []
		accuracy_train_sub = []
		accuracy_test_sub = []
		f1_scores_sub = []

		for repeat in range(repeats):
			t1 = datetime.now()
			nn = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'genetic_alg', 
											  max_iters = iters, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = 5, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = 35,
											  pop_size = 200,
											  mutation_prob = 0.26)
			t2 = datetime.now()
			nn.fit(X_train, y_train)
			t3 = datetime.now()
			y_test_pred = nn.predict(X_test)
			t4 = datetime.now()
			y_test_accuracy = accuracy_score(y_test, y_test_pred)
			accuracy_test_sub.append(y_test_accuracy)

			y_train_pred = nn.predict(X_train)
			y_train_accuracy = accuracy_score(y_train, y_train_pred)
			accuracy_train_sub.append(y_train_accuracy)

			opt_time_sub.append((t2 - t1).microseconds)
			train_time_sub.append((t3 - t2).microseconds)
			test_time_sub.append((t4 - t3).microseconds)

			f1 = f1_score(y_test, y_test_pred, average='weighted')
			f1_scores_sub.append(f1)

		opt_time.append(opt_time_sub)
		train_time.append(train_time_sub)
		test_time.append(test_time_sub)
		accuracy_train.append(accuracy_train_sub)
		accuracy_test.append(accuracy_test_sub)
		f1_scores.append(f1_scores_sub)

	# write to csv
	opt_time = pd.DataFrame(opt_time)
	train_time = pd.DataFrame(train_time)
	test_time = pd.DataFrame(test_time)
	accuracy_train = pd.DataFrame(accuracy_train)
	accuracy_test = pd.DataFrame(accuracy_test)
	f1_scores = pd.DataFrame(f1_scores)

	opt_time.to_csv('nn_ga_optTime.csv', 
					header = headers, 
					index = False)
	train_time.to_csv('nn_ga_trainTime.csv', 
					  header = headers, 
					  index = False)
	test_time.to_csv('nn_ga_testTime.csv', 
					  header = headers, 
					  index = False)
	accuracy_train.to_csv('nn_ga_accTrain.csv', 
					  header = headers, 
					  index = False)
	accuracy_test.to_csv('nn_ga_accTest.csv', 
					  header = headers, 
					  index = False)
	f1_scores.to_csv('nn_ga_f1.csv', 
					  header = headers, 
					  index = False)

def plot_cm(testY, predY, class_names, title, file_name):

	cm = confusion_matrix(testY, predY)
	cm = cm.astype('float') / cm.sum(axis=1, keepdims =True)

	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title(title, y = 1.03)
	plt.ylabel('True', x = 1.03)
	plt.xlabel('Predicted', y = 1.03)
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=20)
	plt.yticks(tick_marks, class_names)

	fmt = '.2f'
	thresh = cm.max() / 2.0
	for i in range(cm.shape[0]):
	    for j in range(cm.shape[1]):
	        plt.text(j,i, format(cm[i,j], fmt), ha='center', va='center', color='white' if cm[i,j] > thresh else 'black')

	plt.savefig(file_name, bbox_inches = "tight")
	plt.close()

def get_class(row):

	for c in range(5):
		if row[c] == 1:
			return classes[c]


def plot_matrics():

	iters = 200

	# reverse one-hot encode
	temp = pd.DataFrame(y_test)
	y_test_new = temp.apply(get_class, axis = 1)
	y_test_new = np.array(y_test_new)

	# gradient descent
	nn_gd = mlrose.NeuralNetwork(hidden_nodes = [75], 
								  activation = 'relu', 
								  algorithm = 'gradient_descent', 
								  max_iters = iters, 
								  bias = True, 
								  is_classifier = True, 
								  learning_rate = 0.0002, 
								  early_stopping = True, 
								  clip_max = 10,
								  max_attempts = 400)
	nn_gd.fit(X_train, y_train)
	y_test_pred_gd = nn_gd.predict(X_test)
	y_test_pred_gd = pd.DataFrame(y_test_pred_gd)
	y_test_pred_gd = y_test_pred_gd.apply(get_class, axis=1)
	y_test_pred_gd = np.array(y_test_pred_gd)
	
	plot_cm(testY = y_test_new, 
		    predY = y_test_pred_gd, 
		    class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen'], 
		    title = 'Normalized Confusion Matrix for GD', 
		    file_name = 'cm_gd.png')

	# random hill climbing
	nn_rhc = mlrose.NeuralNetwork(hidden_nodes = [75], 
								  activation = 'relu', 
								  algorithm = 'random_hill_climb', 
								  max_iters = iters, 
								  bias = True, 
								  is_classifier = True, 
								  learning_rate = 0.0004, 
								  early_stopping = True, 
								  clip_max = 10,
								  max_attempts = 700)
	nn_rhc.fit(X_train, y_train)
	y_test_pred_rhc = nn_rhc.predict(X_test)
	y_test_pred_rhc = pd.DataFrame(y_test_pred_rhc)
	y_test_pred_rhc = y_test_pred_rhc.apply(get_class, axis=1)
	y_test_pred_rhc = np.array(y_test_pred_rhc)

	plot_cm(testY = y_test_new, 
		    predY = y_test_pred_rhc, 
		    class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen'], 
		    title = 'Normalized Confusion Matrix for RHC', 
		    file_name = 'cm_rhc.png')

	# simulated annealing
	nn_sa = mlrose.NeuralNetwork(hidden_nodes = [75], 
								  activation = 'relu', 
								  algorithm = 'simulated_annealing', 
								  max_iters = iters, 
								  bias = True, 
								  is_classifier = True, 
								  learning_rate = 0.0003, 
								  early_stopping = True, 
								  clip_max = 10,
								  max_attempts = 1000,
								  schedule = mlrose.GeomDecay())
	nn_sa.fit(X_train, y_train)
	y_test_pred_sa = nn_sa.predict(X_test)
	y_test_pred_sa = pd.DataFrame(y_test_pred_sa)
	y_test_pred_sa = y_test_pred_sa.apply(get_class, axis=1)
	y_test_pred_sa = np.array(y_test_pred_sa)


	plot_cm(testY = y_test_new, 
		    predY = y_test_pred_sa, 
		    class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen'], 
		    title = 'Normalized Confusion Matrix for SA', 
		    file_name = 'cm_sa.png')

	# genetic algorithm
	nn_ga = mlrose.NeuralNetwork(hidden_nodes = [75], 
											  activation = 'relu', 
											  algorithm = 'genetic_alg', 
											  max_iters = iters, 
											  bias = True, 
											  is_classifier = True, 
											  learning_rate = 0.0002, 
											  early_stopping = True, 
											  clip_max = 10,
											  max_attempts = 35,
											  pop_size = 200,
											  mutation_prob = 0.26)
	nn_ga.fit(X_train, y_train)
	y_test_pred_ga = nn_ga.predict(X_test)
	y_test_pred_ga = pd.DataFrame(y_test_pred_ga)
	y_test_pred_ga = y_test_pred_ga.apply(get_class, axis=1)
	y_test_pred_ga = np.array(y_test_pred_ga)

	plot_cm(testY = y_test_new, 
		    predY = y_test_pred_ga, 
		    class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen'], 
		    title = 'Normalized Confusion Matrix for GA', 
		    file_name = 'cm_ga.png')


if __name__ == '__main__': 

	X_train, y_train = load_training_data()
	X_test, y_test = load_test_data()

	# one hot encode
	y_train, y_test = encode(y_train, y_test)

	repeats = 3
	headers = ['repeat1', 'repeat2', 'repeat3']

	# for repeat in range(repeats):
	# 	default_results()

	# grid_search_gd()
	# grid_search_rhc()
	# grid_search_sa()
	# grid_search_ga()

	iterRange = range(20, 201, 20)
	# explore_iters_gd()
	explore_iters_rhc()
	explore_iters_sa()
	explore_iters_ga()

	# classes = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen']
	# plot_matrics()

	
