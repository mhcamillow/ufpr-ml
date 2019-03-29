#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
from sklearn.utils import shuffle

def main(train_data, test_data):

	train_features, train_labels = load_svmlight_file(train_data)
	test_features, test_labels = load_svmlight_file(test_data)

	train_features = shuffle(train_features, random_state=5)
	train_labels = shuffle(train_labels, random_state=5)

	partition_data_in_x_lines = 1000
	data_size = len(train_labels)
	number_of_batches = data_size / partition_data_in_x_lines

	print "Number of examples: " + str(data_size)
	print "Batches size: " + str(partition_data_in_x_lines)
	print "Batch number: " + str(number_of_batches)

	for i in range(number_of_batches):
		batch_end = (i + 1) * partition_data_in_x_lines
		batch_labels = train_labels[:batch_end]
		batch_features = train_features.toarray()[:batch_end]
		t1 = time.time()
		run_test(batch_features, batch_labels, test_features.toarray(), test_labels)
		print str(time.time() - t1) + ' seconds.'

	# t1 = time.time()
	# run_test(train_features, train_labels, test_features, test_labels)
	# print str(time.time() - t1) + ' seconds.'

def run_test(train_features, train_labels, test_features, test_labels):

	print 'Runnint with ' + str(len(train_labels)) + ' training examples.'
	neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
	neigh.fit(train_features, train_labels)
	y_pred = neigh.predict(test_features)
	
	print neigh.score(test_features, test_labels)
	cm = confusion_matrix(test_labels, y_pred)
	print cm

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: knn.py <train_data> <test_data>")

	main(sys.argv[1], sys.argv[2])