"""
               Content Based Algorithms
===================================================

This module implements content base recommendation algorithms:
    - TFIDF
    - BM25
    - JACCARD

"""

# Author: Caleb De La Cruz P. <cdelacru>


import logging
from CB.CBAlgorithm import CBAlgorithm
from Util import Field

logger = logging.getLogger(__name__)


class CBRecommender(object):
    def train(self, data, algorithm):
        '''
        Given a dataset:
        - Index it
        - Calculate the similarity score of each pair of items using the specified algorithm
        :param data: List of strings
        :param algorithm: [tfidf, bm25f, jaccard]
        :return: Sparse matrix NxN with the similarity score of every pair of items
        '''
        if not isinstance(data, list):
            raise AttributeError("The parameter data should be a list of string")
        # if not isinstance(algorithm, CBAlgorithm):
        #     raise AttributeError("The parameter algorithm should be an instance of CBAlgorithm or its childs")

        index = algorithm.index(data)
        score = algorithm.similarity(index)

        return score

    def train_several_fields(self, fields, algorithm, dataset, weigths):
        '''
        Index the dataset for a set of fields
         - calculate the similarity score of each pair of items using the specified algorithm for each field
         - Aggregate the score using a weighted average

         IMPORTANT: The length of each dataset should be the same
        :param field: String [Title, Plot, ]
        :param algorithm: [tfidf, bm25f, jaccard]
        :return: Sparse matrix NxN with the similarity score of every pair of items
        '''
        if not isinstance(fields, list):
            raise AttributeError("The parameter fields should be a list of Field values")
        if not isinstance(weigths, list):
            raise AttributeError("The parameter weigths should be a list of Field values")

        if len(fields) == 0:
            return None

        if not isinstance(fields[0], Field):
            raise AttributeError("The parameter fields should be a list of Field values")
        if not isinstance(algorithm, CBAlgorithm):
            raise AttributeError("The parameter algorithm should be an instance of CBAlgorithm or its childs")
        if len(fields) != len(weigths):
            raise AttributeError("The number of fields should be equal to the number of weights")

        scores = []
        for field in fields:
            data = dataset.get_data(field)
            scores.append(self.train(data, algorithm))

        # Try to aggregate the different scores
        aggregated_score = weigths[0] * scores[0]
        i = 1
        for score in scores[1:]:
            if aggregated_score.shape != scores[i].shape:
                raise IndexError("The field No. %d only have %d items whereas the other fields have %d items" % \
                                 (i + 1, len(aggregated_score), len(score)))
            aggregated_score += weigths[i] * scores[i]
            i += 1

        return aggregated_score

    def save(self, field, algorithm):
        '''
        Given a matrix of scores save/update every pair of recommendations to a persistent storage, using the appropiate
        column for the given fields and algorithms
        :param scores:
        :return:
        '''
        # TODO: Save method
        raise NotImplementedError("#TODO")
