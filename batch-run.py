#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
from sklearn.utils import shuffle
from time import gmtime, strftime

file_to_write = strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".txt"

def main(train_data, test_data):

	# classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
	# classifier = GaussianNB()
	# classifier = LinearDiscriminantAnalysis()
	classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

	train_features, train_labels = load_svmlight_file(train_data)
	test_features, test_labels = load_svmlight_file(test_data)

	train_features = shuffle(train_features, random_state=5)
	train_labels = shuffle(train_labels, random_state=5)

	partition_data_in_x_lines = 1000
	data_size = len(train_labels)
	number_of_batches = math.ceil(data_size / partition_data_in_x_lines)

	for i in range(number_of_batches):
		batch_end = (i + 1) * partition_data_in_x_lines
		batch_labels = train_labels[:batch_end]
		batch_features = train_features.toarray()[:batch_end]
		run_test(classifier, batch_features, batch_labels, test_features.toarray(), test_labels)

def run_test(classifier, train_features, train_labels, test_features, test_labels):
	t1 = time.time()
	classifier.fit(train_features, train_labels)
	y_pred = classifier.predict(test_features)
	total_time = round(time.time() - t1, 2)
	cm = confusion_matrix(test_labels, y_pred)
	size_training_batch = len(train_labels)
	score = classifier.score(test_features, test_labels)
	tp = cm[0][0]
	tn = cm[1][1]
	fp = cm[0][1]
	fn = cm[1][0]
	acc = round((tp + tn) / (tp + tn + fp + fn), 2)
	precision = round((tp) / (tp + fp), 2)
	recall = round((tp) / (tp + fn), 2)
	f1score = round(2 * ((precision * recall) / (precision + recall)), 2)
	str_to_print = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}".format(size_training_batch, score, total_time, tp, tn, fp, fn, acc, precision, recall, f1score)
	fout = open(file_to_write, "a")
	print(str_to_print)
	fout.write(str_to_print)
	fout.write("\n")
	fout.close

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: knn.py <train_data> <test_data>")

	main(sys.argv[1], sys.argv[2])