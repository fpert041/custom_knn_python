#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:00:48 2017

@author: pesa
"""

import numpy as np
from sklearn import datasets

iris = datasets.load_iris() # load data set


print(iris.DESCR)

X = iris.data #measurements [4 bits of measurements in this case]
y = iris.target # labels / classes

mySeed = 12345678
np.random.seed(mySeed)

# add noise
X = X + np.random.normal(0,0.5,X.shape)

# randomly permutated ascending sequence of integers
indeces = np.random.permutation(X.shape[0])

# create 2 arrays of indeces (one to pick training data, one to pick test data)
bins = np.array_split(indeces, 5)
foldTrainInd = np.concatenate((bins[0], bins[1], bins[2], bins[3]))
foldTestInd = bins[0]

# define methods to get distance between two data-points

def euclideanDistance(in1, in2):
    result=0
    for i in range(0, len(in1)):
        result += (in1[i] - in2[i])**2
    return result

def trueEuclideanDistance(in1, in2):
    result=0
    for i in range(0, len(in1)):
        result += (in1[i] - in2[i])**2
    return np.sqrt(result)

def absDist(in1, in2):
    result=0
    for i in range(0, len(in1)):
        result += abs(in1[i] - in2[i])
    return np.sqrt(result)


# define method to get closest neighbours in the trainig set
    # args: data_test_point, training_set, num_neighbours
    
def getNeighbours(x_, inX, n):
    #X=np.array(inX)
    distances = np.zeros(len(inX)) #create an empty array of 0s
    for i in range (0, len(distances)):
        distances[i] = euclideanDistance(x_, inX[i]) #fill it 
                                            # with distances between the test_point x_
                                            # and the data_points in training set
    sortedIndeces = np.argsort(distances) # get the indeces of distances
                                                # in order of distance
    return sortedIndeces[:n]
 

from collections import Counter

# def. function that takes in a training set of labels
# and returns the most frequently occurring label in such set         
def assignLabel(nLabels):
    data = Counter(nLabels)
    mode = data.most_common(1)[0][0]
    return mode


correct = 0
neighbours_number = 10

for i in foldTestInd: # for all the data points in our TEST dataset
    x_ = X[i] # test point
    y_ = y[i] # true label 
    
    # get closest neighbours indeces
    nIndeces = getNeighbours(x_, X[foldTrainInd], neighbours_number)
    
    # predict what the testing datapoint is based on closest neighbours training labels
    prediction = assignLabel( y[foldTrainInd][nIndeces] )
    
    if(prediction == y_):
        correct += 1
    

accuracy = correct / len(foldTestInd)
print(accuracy)


# benchmark --------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
    

knn = KNeighborsClassifier(neighbours_number, metric = 'euclidean')
knn.fit(X[foldTrainInd], y[foldTrainInd])

y_pred = knn.predict(X[foldTestInd])

print(accuracy_score(y[foldTestInd], y_pred))

# custom class ---------

from MyKNN import MyKnn
from MyKNN import MyMetrics as mm

myKnn = MyKnn(neighbours_number, metric = 'euclidean')
myKnn.train(X[foldTrainInd], y[foldTrainInd])

my_y_pred = myKnn.predict(X[foldTestInd])

print(mm.accuracy_score(y[foldTestInd], y_pred))