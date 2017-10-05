"""
     JACCARD Content Based Algorithm
===================================================

This implementation uses JACCARD to measure similarity between items. It uses a vectorized approach to calculating
 the JACCARD score in order to improve performance


"""

# Author: Caleb De La Cruz P. <cdelacru>


import logging
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

from .CBAlgorithm import CBAlgorithm
#import dask.array as da
#from chest import Chest

logger = logging.getLogger(__name__)


class CBAlgorithmJACCARD(CBAlgorithm):

    def __init__(self):
        self.__name__ = 'JACCARD'

    def index(self, data):
        '''
        Index the dataset using TFIDF as score
        :param data: Array of strings
        :return: Sparse matrix NxM where N is the same length of data and M is the number of features
        '''
        data = super(CBAlgorithmJACCARD, self).index(data)

        t0 = time()
        self.vectorizer = CountVectorizer(max_df=0.5, stop_words='english')
        self.indexed = self.vectorizer.fit_transform(data)
        duration = time() - t0
        logger.debug("n_samples: %d, n_features: %d" % self.indexed.shape)
        logger.debug("duration: %d" % duration)
        return self.indexed

    def similarity(self, index=None):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.indexed
        super(CBAlgorithmJACCARD, self).similarity(index)

        t0 = time()
        # Transform index to be a binary index
        binary_index = index
        binary_index[binary_index != 0] = 1

        # Calculate intersection in a vectorized way
        intersection = binary_index.dot(binary_index.T)

        return intersection

        # Calculate union in a vectorized way
        union = self.vectorized_union(intersection, binary_index)

        # Divide intersection over union to get JACCARD score
        result = union.copy()
        result.data = np.divide(intersection.data, union.data)

        duration = time() - t0
        logger.debug("n_samples: %d, n_related_samples: %d" % result.shape)
        logger.debug("duration: %d" % duration)

        # TODO: Check the effect of using only the intersection score instead of the JACCARD score
        return  result

    def vectorized_union(self, intersection, data):

        # Step 1: Aggregate each row
        row = np.ravel(data.sum(1))

        # Step 2: Append a row/column full of ones
        ones = np.ones(row.shape)
        matrix_1 = np.append([row], [ones], axis=0)
        matrix_2 = np.append([ones.T], [row.T], axis=0).T

        # Step 3: Calculate the sum of each possible combination
        matrix_1 = da.from_array(matrix_1, chunks=(1000))
        matrix_2 = da.from_array(matrix_2, chunks=(1000))

        union = matrix_2.dot(matrix_1)
        union = union.compute()
        logger.debug("Dot Product")

        # cache = Chest(path='c:/temp', available_memory=13e9)
        # union = matrix_2.dot(matrix_1)#.compute(cache=cache)

        # Step 4: Remove intersection
        # Step 4.1: 0 everything that does not have an intersection
        #           This step is not for UNION, but in JACCARD we dont care about
        #           Cells without intersection

        indexes = intersection.nonzero()
        logger.debug("Data")
        sparse_union = csr_matrix((union[indexes], indexes), shape=intersection.shape)
        logger.debug("Sparse")
        # Step 4.2: Remove Intersection
        sparse_union.data = sparse_union.data - intersection.data
        logger.debug(sparse_union.data[0:10])
        return sparse_union

