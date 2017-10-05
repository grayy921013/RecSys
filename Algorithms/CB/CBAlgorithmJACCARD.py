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

    def similarity(self, index=None, epsilon=1):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items

        reference: https://stackoverflow.com/a/32885931/1354478
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.indexed
        super(CBAlgorithmJACCARD, self).similarity(index)

        t0 = time()
        
        assert(0 < epsilon <= 1)
        
        # Transform index to be a binary index
        csr = index
        csr = csr.astype(bool).astype(float)

        csr_rownnz = csr.getnnz(axis=1)
        intrsct = csr.dot(csr.T)
        
        nnz_i = np.repeat(csr_rownnz, intrsct.getnnz(axis=1))
        unions = nnz_i + csr_rownnz[intrsct.indices] - intrsct.data
        dists = 1.0 - intrsct.data / unions
        
        mask = (dists > 0) | (dists <= epsilon)
        data = dists[mask]
        # TODO: Fix force to mantain same # of references with the
        #       the other algorithms to avoid having to merge
        data[data == 0] = 1e-10
        indices = intrsct.indices[mask]
        
        idx = intrsct.indptr[:-1]
        i = -1
        while idx[i] == len(mask):
            idx[i] -= 1
            i -= 1
        rownnz = np.add.reduceat(mask, idx)
        # TODO: Fix patch that corrects a numpy error on the add.reduceat
        #       function when the indexes are the same instead of having a
        #       numpy is adding a 1, which leds to overflow
        idx1 = intrsct.indptr[:-1]
        idx2 = intrsct.indptr[1:]
        idx3 = (idx1-idx2) == 0
        rownnz[idx3] = 0
        indptr = np.r_[0, np.cumsum(rownnz)]
        
        out = csr_matrix((data, indices, indptr), intrsct.shape)

        # Zero out redundant scores
        # Ex. The movies (2,1) and (1,2) will have the same score
        #     Thus without loosing generality 
        #     we will only save the pairs where m1 < m2
        lower_triangle_idx = np.tril_indices(out.shape[0])
        out[lower_triangle_idx] = 0
        out.eliminate_zeros()

        return out
