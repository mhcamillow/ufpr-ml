#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
import time
from sklearn.model_selection import train_test_split
import CamilloKNN
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl

def main(data):

    # loads data
    print "Loading data..."
    X_data, y_data = load_svmlight_file(data)
    # splits data
    print "Spliting data..."
    X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.2, random_state = 5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # cria um kNN
    neigh = CamilloKNN.CamilloKNN(n_neighbors=3)

    print 'Fitting knn'
    t1 = time.time()
    neigh.fit(X_train, y_train)
    print 'Time to fit: ' + str(time.time() - t1)

    # predicao do classificador
    print 'Predicting...'
    t1 = time.time()
    y_pred = neigh.predict(X_test)
    print 'Time to predict (seconds): ' + str(time.time() - t1)

    # mostra o resultado do classificador na base de teste
    # print neigh.score(X_test, y_test)

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
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

    # pl.matshow(cm)
    # pl.colorbar()
    # pl.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: knn.py <data>")

    main(sys.argv[1])


