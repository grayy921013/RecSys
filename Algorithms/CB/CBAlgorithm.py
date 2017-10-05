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

    def __init__(self):
        self.ids = None
        self.similarity_matrix = None

    def __str__(self):
        return self.__name__
        
    @abstractmethod
    def index(self, data):
        '''
        Index the dataset
        :param data: Array of strings
        :return: Sparse matrix NxM where N is the same length of data and M is the number of features
        '''
        if not isinstance(data, list):
            raise AttributeError("The parameter data should be an array of strings")

        matrix = np.array(data)
        self.ids = matrix[:, 0].astype(int)
        values = matrix[:, 1]

        return values.tolist()

    @abstractmethod
    def similarity(self, index):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if not isinstance(index, np.ndarray) and not isinstance(index, sp.sparse.spmatrix):
            logger.error(type(index))
            raise AttributeError("The parameter index should be an numpy matrix")

    def add_similarity(self, index=None):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.indexed
        if index is None:
            raise AttributeError("You should run the index function before calculating the similarity")

        t0 = time()

        binary_index = index.copy()
        binary_index[binary_index != 0] = 1

        score = index.dot(binary_index.T)
        duration = time() - t0
        # TODO: Figure out how to 0 out main diagonal on sparse matrix
        # np.fill_diagonal(score, 0)
        logger.debug("n_samples: %d, n_related_samples: %d" % score.shape)
        logger.debug("duration: %f\n" % duration)

        self.similarity_matrix = score

        return score

    def dot_product_similarity(self, index=None):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.indexed
        if index is None:
            raise AttributeError("You should run the index function before calculating the similarity")

        t0 = time()
        score = index.dot(index.T)
        duration = time() - t0
        # Zero out redundant scores
        # Ex. The movies (2,1) and (1,2) will have the same score
        #     Thus without loosing generality 
        #     we will only save the pairs where m1 < m2
        lower_triangle_idx = np.tril_indices(score.shape[0])
        score[lower_triangle_idx] = 0
        score.eliminate_zeros()

        # TODO: Figure out how to 0 out main diagonal on sparse matrix
        # np.fill_diagonal(score, 0)
        logger.debug("n_samples: %d, n_related_samples: %d" % score.shape)
        logger.debug("duration: %f\n" % duration)

        self.similarity_matrix = score

        return score

    def ranking(self, similarity_matrix=None, rank_length=21, flag=False):
        # TODO: remove ranking itself

        # Reference:
        # https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
        if similarity_matrix is None:
            raise AttributeError("You should run the similarity function before calculating a ranking")

        n_movies = similarity_matrix.shape[0]

        top = []
        j = 0
        for i in xrange(n_movies):
            if i % 10000 == 0:
                logger.debug('ranking at position %d' % i)

            # Only get the data
            related_movies_scores_for_i = similarity_matrix[i, :].data
            related_movies_id_for_i = similarity_matrix[i, :].indices

            # Look the top N values in that N
            if len(related_movies_id_for_i) < rank_length:
                # If it already only has fewer possible similars, just pick the whole set
                top_n_ids = related_movies_id_for_i
                top_n_scores = related_movies_scores_for_i
                j += 1
            else:
                # Split the whole thing
                top_n = np.argpartition(related_movies_scores_for_i, -rank_length, axis=0)[-rank_length:]
                top_n_ids = related_movies_id_for_i[top_n]
                top_n_scores = related_movies_scores_for_i[top_n]

            # Transform Index to DB ids
            r = set()
            for i in top_n_ids:
                r.add(self.ids[i])
            # top.append(r)

            # TODO: Check if I really should have r as a set
            r2 = zip(list(r), top_n_scores)
            top.append(r2)

        logger.debug('Movies Processed: %d Movies without enough Related Movies: %d' % (len(top), j))

        if flag:
            top = zip(list(self.ids), top)
        return top

    def score(self, similarity_matrix, test_data):

        counter_tp = 0
        counter_fp = 0

        top = self.ranking(similarity_matrix)

        for record in test_data:
            movie_id1, movie_id2, positive = record
            index1 = np.argmax(self.ids == movie_id1)
            index2 = np.argmax(self.ids == movie_id2)

            if positive:
                if movie_id2 in top[index1]:
                    counter_tp += 0.5

                if movie_id1 in top[index2]:
                    counter_tp += 0.5
            else:
                if movie_id2 in top[index1]:
                    counter_fp += 0.5

                if movie_id1 in top[index2]:
                    counter_fp += 0.5

        logger.debug('TP %d FP %d Total %d' % (counter_tp, counter_fp, len(test_data)))

        return counter_tp, counter_fp

    def compare(self, top, baseline, baseline_ids):

        size_ids = len(self.ids)

        if size_ids != len(top):
            raise AttributeError()

        related_movies_set = [None] * size_ids
        idx = 0
        j = 0

        for i in self.ids:
            i = int(i)
            baseline_idx = np.argmax(baseline_ids ==  i)

            if baseline_idx:
                related_movies_set[idx] = baseline[baseline_idx]
            else:
                related_movies_set[idx] = set()
                j += 1
            idx += 1

        logger.debug('Movies %d Skipped %d' % (size_ids, j))

        counter = 0
        total = 0
        for i in xrange(len(related_movies_set)):
            counter += len(related_movies_set[i].intersection(top[i]))
            total += len(related_movies_set[i])

        if total == 0:
            return -1

        PRECISSION = counter / float(total)

        # related_movies_set
        return PRECISSION

    def destroy(self):
        self.ids = None
        self.vectorizer = None
        self.index = None
        self.similarity_matrix = None
