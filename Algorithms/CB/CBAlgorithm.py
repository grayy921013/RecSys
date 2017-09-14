"""
     Content Based Algorithms Base Class
===================================================

All CBAlgorithms should inherit from this class and included the methods here defined

"""

# Author: Caleb De La Cruz P. <cdelacru>


import logging
from time import time
import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)


class CBAlgorithm(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def index(self, data):
        '''
        Index the dataset
        :param data: Array of strings
        :return: Sparse matrix NxM where N is the same length of data and M is the number of features
        '''
        if not isinstance(data, list):
            raise AttributeError("The parameter data should be an array of strings")

    @abstractmethod
    def similarity(self, index):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if not isinstance(index, np.ndarray) and not isinstance(index, sp.sparse.spmatrix):
            print type(index)
            raise AttributeError("The parameter index should be an numpy matrix")

    def dot_product_similarity(self, index=None):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.index
        t0 = time()
        score = index.dot(index.T)
        duration = time() - t0
        # TODO: Figure out how to 0 out main diagonal on sparse matrix
        # np.fill_diagonal(score, 0)
        logger.info("n_samples: %d, n_related_samples: %d" % score.shape)
        logger.info("duration: %f\n" % duration)

        return score
