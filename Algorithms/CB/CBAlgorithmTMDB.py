"""
     TFIDIF Content Based Algorithm
===================================================

This implementation uses TFIDF to measure similarity between items. It does this using the Scikit-Learn
TfidfVectorizer. Using its fit_transform function.

"""

# Author: Caleb De La Cruz P. <delacruzp>


import logging
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

from Data.get_data import get_related_tmdb
from .CBAlgorithm import CBAlgorithm
import numpy as np


logger = logging.getLogger(__name__)


class CBAlgorithmTMDB(CBAlgorithm):

    def __init__(self):
        self.__name__ = 'TMDB'

    def index(self, data):
        '''
        Index the dataset using TFIDF as score
        :param data: Array of strings
        :return: Sparse matrix NxM where N is the same length of data and M is the number of features
        '''
        pass
        # raise NotImplementedError()

    def similarity(self, index=None):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        pass
        # raise NotImplementedError()

    def ranking(self, similarity_matrix=None, rank_length=21, flag=True):
        top, ids = get_related_tmdb(rank_length)
        self.ids = np.array(ids)
        if flag:
            top = zip(ids, top)
        return top