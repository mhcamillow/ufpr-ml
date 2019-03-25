import sys
import math
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

class CamilloKNN():

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, train_examples, train_labels):
        self.train_examples = train_examples
        self.train_labels = train_labels
        self.number_of_classes = len(set(train_labels))
    
    def predict(self, test_data):
        predicted = []
        t1_total = time.time()
        time_calculating_distances = 0
        time_getting_closest_elements = 0
        for test in test_data:
            t1 = time.time()
            distances = self.get_distances(test)
            time_calculating_distances += time.time() - t1

            t1 = time.time()
            closest_k_elements = self.get_first_n_elements(distances, self.n_neighbors)
            time_getting_closest_elements += time.time() - t1

            guess_label = self.get_closest_class(closest_k_elements)
            predicted.append(guess_label)
        
        total_time = time.time() - t1_total
        print 'Time spent calculating distances: ' + str(round(time_calculating_distances, 2)) + '(' + str(round(time_calculating_distances / total_time, 2)) + ')'
        print 'Time spent getting closest elements: ' + str(round(time_getting_closest_elements, 2)) + '(' + str(round(time_getting_closest_elements / total_time, 2)) + ')'
        print 'Total time: ' + str(total_time)
        return predicted
    
    def get_distances(self, test):
        distances = []

        for train in self.train_examples:
            dif = np.subtract(train, test)
            pow_dif = np.power(dif, 2)
            sum_pow_dif = sum(pow_dif)
            sqrt_sum_pow_dif = math.sqrt(sum_pow_dif) 
            distances.append(sqrt_sum_pow_dif)
        
        return distances
    
    def get_first_n_elements(self, array, n):
        return sorted(range(len(array)), key=lambda k: array[k])[:n]

    def get_closest_class(self, elements):
        res = [0] * self.number_of_classes

        for k_elements in elements:
            res[int(self.train_labels[k_elements])] += 1
        guess = sorted(range(len(res)), key=lambda k: res[k])[self.number_of_classes - 1]

        return guess