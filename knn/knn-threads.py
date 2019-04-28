#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
import time
from sklearn.model_selection import train_test_split
import CamilloKNNThreads
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing

def main(train_data, test_data, k):
    train_features, train_classes = load_svmlight_file(train_data)
    test_features, test_classes = load_svmlight_file(test_data)

    neigh = CamilloKNNThreads.CamilloKNN(n_neighbors=k)
    neigh.fit(train_features.toarray(), train_classes)

    t1 = time.time()
    test_predictions = neigh.predict(test_features.toarray())
    print('Time to predict: ' + str(round(time.time() - t1, 4)) + " seconds.")
    
    cm = confusion_matrix(test_classes, test_predictions)
    acc = neigh.score(cm)
    print("Score: {0}".format(acc))
    print('Confusion matrix:')
    print(cm)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Use: knn.py <train_data> <test_data> <k>")

    main(sys.argv[1], sys.argv[2], sys.argv[3])


