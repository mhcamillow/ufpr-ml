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

def main(train_data, test_data):

    # loads data
    print "Loading data..."
    train_features, train_classes = load_svmlight_file(train_data)
    test_features, test_classes = load_svmlight_file(test_data)

    neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    neigh.fit(train_features.toarray(), train_classes)

    print 'Predicting...'
    t1 = time.time()
    test_predictions = neigh.predict(test_features.toarray())
    print 'Time to predict (seconds): ' + str(time.time() - t1)

    cm = confusion_matrix(test_classes, test_predictions)
    print 'Confusion matrix:'
    print cm

    tp = float(cm[0][0])
    tn = float(cm[1][1])
    fp = float(cm[0][1])
    fn = float(cm[1][0])

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f1score = 2 * ((precision * recall) / (precision + recall))
    print 'Accuracy: ' + str(acc)
    print 'Precision: ' + str(precision)
    print 'Recall: ' + str(recall)
    print 'F1 Score: ' + str(f1score)

    pl.matshow(cm)
    pl.colorbar()
    pl.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: knn.py <train_data> <test_data>")

    main(sys.argv[1], sys.argv[2])


