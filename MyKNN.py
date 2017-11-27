#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:16:09 2017

@author: pesa
"""

from collections import Counter
import numpy as np

class MyMetrics:
    
    def accuracy_score(true_labesl_y, predicted_label_y):
        correct = 0
        for i in range(0, len(true_labesl_y)):
            if true_labesl_y[i] == predicted_label_y[i] :
                correct += 1
                
        accuracy = correct / len(true_labesl_y)
        return accuracy

class MyKnn:
    
    # --------- variables ---------
    
    training_set_X = []
    training_set_y = []
    num_neighbours = 1
    metric = 'euclidean'
    def getDistance() :
        return 0
    
    # --------- helper methods ---------
    
    # define methods to get distance between two data-points
    def euclideanDistance(self, in1, in2):
        result=0
        for i in range(0, len(in1)):
            result += (in1[i] - in2[i])**2
        return result

    def trueEuclideanDistance(self, in1, in2):
        result=0
        for i in range(0, len(in1)):
            result += (in1[i] - in2[i])**2
        return np.sqrt(result)

    def absDist(self, in1, in2):
        result=0
        for i in range(0, len(in1)):
            result += abs(in1[i] - in2[i])
            return np.sqrt(result)


    # define method to get closest neighbours in the trainig set
    # args: data_test_point, training_set, num_neighbours
    
    def getNeighbours(self, x_, inX, n):
        #X=np.array(inX)
        distances = np.zeros(len(inX)) #create an empty array of 0s
        for i in range (0, len(distances)):
            distances[i] = self.euclideanDistance(x_, inX[i]) #fill it 
                                            # with distances between the test_point x_
                                            # and the data_points in training set
            sortedIndeces = np.argsort(distances) # get the indeces of distances
                                                # in order of distance
        return sortedIndeces[:n]
 

    

    # def. function that takes in a training set of labels
    # and returns the most frequently occurring label in such set         
    def assignLabel(self, nLabels):
        data = Counter(nLabels)
        mode = data.most_common(1)[0][0]
        return mode
    
     # --------- constructor / initialiser ---------
     
    def __init__(self, neighbours_number = 1, metric = 'euclidean'):
         self.num_neighbours = neighbours_number
         if metric == 'euclidean':
             self.getDistance = self.euclideanDistance
             self.metric = metric
         else:
             if metric == 'true_euclidean':
                 self.getDistance = self.trueEuclideanDistance
                 self.metric = metric
             else:
                 if metric == 'absolute':
                     self.getDistance = self.absDist
                     self.metric = metric
                 else:
                     print('metric not recognised: defaulting to euclidean')
                
    
    # --------- setter method ---------
    
    def train(self, in_train_dataset_X, in_train_dataset_y):
        self.training_set_X = in_train_dataset_X
        self.training_set_y = in_train_dataset_y

    # --------- ML prediction method ---------
    
    def predict(self, test_set_x):
        
        len_set = len(test_set_x)
        predictions = np.zeros(len_set)
        for i in range(0, len_set): # for all the data points in our TEST dataset
            x_ = test_set_x[i] # test point
    
            # get closest neighbours indeces
            nIndeces = self.getNeighbours(x_, self.training_set_X, self.num_neighbours)
    
            # predict what the testing datapoint is based on closest neighbours training labels
            predictions[i] = self.assignLabel( self.training_set_y[nIndeces] )
        return predictions