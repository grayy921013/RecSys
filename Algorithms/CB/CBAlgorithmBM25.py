"""
     BM25 Content Based Algorithm
===================================================

This implementation uses BM25 to measure similarity between items. It uses a scikit-learn non-approved fork.


"""

# Author: Caleb De La Cruz P. <delacruzp>


import logging
from time import time

from CB.Bm25 import Bm25Vectorizer
from .CBAlgorithm import CBAlgorithm

logger = logging.getLogger(__name__)


class CBAlgorithmBM25(CBAlgorithm):

    def __init__(self):
        self.__name__ = 'BM25'

    def index(self, data):
        '''
        Index the dataset using TFIDF as score
        :param data: Array of strings
        :return: Sparse matrix NxM where N is the same length of data and M is the number of features
        '''
        data = super(CBAlgorithmBM25, self).index(data)

        t0 = time()
        self.vectorizer = Bm25Vectorizer(max_df=0.5, stop_words='english')
        self.indexed = self.vectorizer.fit_transform(data)
        duration = time() - t0
        logger.debug("n_samples: %d, n_features: %d" % self.indexed.shape)
        logger.debug("duration: %d\n" % duration)
        return self.indexed

    def similarity(self, index=None):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.indexed
        super(CBAlgorithmBM25, self).similarity(index)
        return self.dot_product_similarity(index)
