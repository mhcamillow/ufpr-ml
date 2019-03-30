import sys
import math
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import multiprocessing
from multiprocessing import Process, Array

class CamilloKNN():

    def __init__(self, n_neighbors):
        self.n_neighbors = int(n_neighbors)

    def fit(self, train_examples, train_labels):
        self.train_examples = train_examples
        self.train_labels = train_labels
        self.number_of_classes = len(set(train_labels))
    
    def predict(self, test_data):
        predicted = Array('i', range(len(test_data)))
        threads = multiprocessing.cpu_count()
        thread_size = math.ceil(len(test_data) / threads)
        processes = []

        for x in range(threads):
            first = x * thread_size
            last = len(test_data) if (x == threads - 1) else (x + 1) * thread_size
            p = Process(target=self.run_predict, args=(test_data[first:last], first, x, predicted))
            processes.append(p)
            p.start()
            # print("Thread {0}/{1} started.".format(x + 1, threads))

        for process in processes:
            process.join()

        return predicted[0:]

    def run_predict(self, test_data, first, thread_idx, predicted):
        for x, test in enumerate(test_data):
            distances = self.get_distances(test)
            closest_k_elements = self.get_first_n_elements(distances, self.n_neighbors)
            guess_label = self.get_closest_class(closest_k_elements)
            predicted[x + first] = guess_label


    def score(self, confusion_matrix):
        right_guesses = 0
        total = 0
        for i in range(len(confusion_matrix)):
            right_guesses += confusion_matrix[i][i]
        total = sum(sum(confusion_matrix))
        return round(right_guesses / total, 4)

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