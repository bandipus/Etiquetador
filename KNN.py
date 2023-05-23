__authors__ = ['1636012','1637892','1633445']
__group__ = 'DM.18'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from utils import * 

class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        : param train_data : PxMxN matrix corresponding to P greyscale images
        : return : assigns the train set to the matrix self . train_data shaped as PxD
        ( P points in a D dimensional space )
        """
        train_data = train_data.astype(float)
        
        images = train_data.shape[0]
        dims = train_data.shape[1] * train_data.shape[2]
        train_data = train_data.reshape(images,dims)

        self.train_data = train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
        the ij-th entry is the j-th nearest train point to the i-th test point
        """

        test_data = test_data.astype(float)

        images = test_data.shape[0]
        dims = test_data.shape[1] * test_data.shape[2]
        test_data = test_data.reshape(images,dims)
        self.test_data = test_data

        distances = cdist(self.test_data, self.train_data)

        indexs = distances.argsort(axis=1)[:,:k]

        self.neighbors = self.labels[indexs]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        
        most_frequent_neighbors = []
        value_counts=[]

        for neighbor_row in self.neighbors:
            unique, index, counts = np.unique(neighbor_row, return_counts=True, return_index=True)
            value_counts.append(counts)
            values_where_max_starts = np.where(counts == np.amax(counts))

            if values_where_max_starts[0].size == index.size:
                index_where_the_value_appears_firts_time = index.min()
            elif values_where_max_starts[0].size == 1:
                index_where_the_value_appears_firts_time = index[values_where_max_starts[0][0]]
            else:
                index_where_the_value_appears_firts_time = index[values_where_max_starts[0]].min()

            most_frequent_neighbors.append(neighbor_row[index_where_the_value_appears_firts_time])

        return np.array(most_frequent_neighbors)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()