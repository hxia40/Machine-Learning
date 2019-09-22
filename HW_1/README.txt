Machine Learning - 2019 Fall

Assignment 1

Use this Dropbox link to find the code and dataset files, and other associated files needed:


https://www.dropbox.com/sh/tcyqnync50xmp7o/AAAJH-kokjYgRyLp9_gAO7GRa?dl=0

The code file:
	HW1.py

The dataset files:
	fashion-mnist_train_minor.csv, 
	fashion-mnist_test_minor.csv, 
	Epileptic_Seizure_Recognition.csv

Folders:
	figures_and_data
	mnist
	ESR

How to use it:

The code is written in Python 2.7, relevant packages will need to be installed before running the code. 	
The version of these packages are:

	numpy - 1.16.4
	pandas - 0.24.2
	matplotlib - 2.2.4
	scikit-learn - 0.20.4
	Spicy - 1.2.2

To run the code, copy all the above-mentioned dataset files (*.csv), the code (HW1.py), and the folders (i.e. copy everything from the provided dropbox folder) into a same folder. After that, simply run the code. 

Note that it may take up to 24 hours to finish running all the code. 
If needed, the code can be run in sections by choice. Just comment out the sections that are not of interest, only run the rest. Not that you will always need to run the sections "Load and standardize data set MNIST" and "Load and standardize data set ESR" to enable the rest. 

There are sections in total . You can find them after line 762 of the code file. 
The sections are:

	Load and standardize data set MNIST
	MNIST - pre-parameter adjustment
	MNIST - parameter validation curve
	MNIST - post-parameter adjustment
	Load and standardize data set ESR
	ESR - pre-parameter adjustment
	ESR - parameter validation curve
	ESR - post-parameter adjustment
	Inter-model comparison
	Inter-model comparison using default hyperparameters

To facilitate the reviewing process, selected figures and *.txt files generated using the code are included in the sub-folder "figures_and_data". These files are:

	Figure 1.png
	Figure 2.png
	Figure 3a.png
	Figure 3b.png
	Figure 4.png
	Figure 5.png
	Inter_model_comparison.txt
	Inter_model_comparison.txt

The *.txt files are used to create Table 3.