#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing

def main(train_data, test_data):
    train_features, train_classes = load_svmlight_file(train_data)
    test_features, test_classes = load_svmlight_file(test_data)

    gnb = GaussianNB()
    gnb.fit(train_features.toarray(), train_classes)

    t1 = time.time()
    test_predictions = gnb.predict(test_features.toarray())
    print('Time to predict: ' + str(round(time.time() - t1, 4)) + " seconds.")
    
    cm = confusion_matrix(test_classes, test_predictions)
    acc = gnb.score(test_features.toarray(), test_classes)
    print("Score: {0}".format(acc))
    print('Confusion matrix:')
    print(cm)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: knn.py <train_data> <test_data>")

    main(sys.argv[1], sys.argv[2])


