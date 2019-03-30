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

def main(train_data, test_data, k):
    train_features, train_classes = load_svmlight_file(train_data)
    test_features, test_classes = load_svmlight_file(test_data)

    neigh = KNeighborsClassifier(n_neighbors=int(k), metric='euclidean')
    neigh.fit(train_features.toarray(), train_classes)
    t1 = time.time()
    test_predictions = neigh.predict(test_features.toarray())
    print('Time to predict (seconds): ' + str(time.time() - t1))
    acc = neigh.score(test_features.toarray(), test_classes)
    print("Score: {0}".format(acc))
    cm = confusion_matrix(test_classes, test_predictions)
    print('Confusion matrix:')
    print(cm)
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Use: knn.py <train_data> <test_data> <k>")

    main(sys.argv[1], sys.argv[2], sys.argv[3])


