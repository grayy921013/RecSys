"""
     JACCARD Content Based Algorithm
===================================================

This implementation uses JACCARD to measure similarity between items. It uses a vectorized approach to calculating
 the JACCARD score in order to improve performance


"""

# Author: Caleb De La Cruz P. <delacruzp>


import logging
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, squareform
from progressbar import ProgressBar, Bar, Percentage, Timer
import pandas

from .CBAlgorithm import CBAlgorithm

logger = logging.getLogger('root')

def distance_batch(m1,
                   m2,
                   ids_1,
                   ids_2,
                   cap,
                   metric='jaccard'):
    # Calculate Distance
    result = 1-cdist(m1, m2, metric)
    logger.debug('distance')

    result = np.array(result)
    
    # Remove super small values
    result[result < cap] = 0

    # Make the matrix sparse/smaller
    result = csr_matrix(result)
    
    # Return only those values
    rows, cols = result.nonzero()
    rows_movielens = ids_1[rows]
    cols_movielens = ids_2[cols]
    scores = result.data

    # Filter Out similarity between the same movie
    # Ex. Toy Story and Toy Story -_-
    mask = rows_movielens != cols_movielens
    rows_movielens = rows_movielens[mask]
    cols_movielens = cols_movielens[mask]
    scores = scores[mask]

    return rows_movielens, cols_movielens, scores


def get_top_k(rows_movielens, cols_movielens, scores, k):
    # Only save the ids of the movies for the first
    # algorithm, because next one will have the same id
    pre_frame = np.rec.fromarrays((rows_movielens, cols_movielens, scores), \
        names=('id1_id','id2_id','als_cosine'))

    p = pandas.DataFrame(pre_frame)
    
    # Get top K elements for each movieid1 set 1
    p = p \
        .sort_values(by=['id1_id', 'als_cosine'], ascending=False) \
        .groupby('id1_id') \
        .head(k) \
        .reset_index(drop=True)

    return p;

class CBAlgorithmJACCARD(CBAlgorithm):

    def __init__(self):
        self.__name__ = 'JACCARD'

    def index(self, data, max_features=1000):
        '''
        Index the dataset using TFIDF as score
        :param data: Array of strings
        :return: Sparse matrix NxM where N is the same length of data and M is the number of features
        '''
        data = super(CBAlgorithmJACCARD, self).index(data)

        t0 = time()
        self.vectorizer = CountVectorizer(max_df=0.5,
            max_features=max_features,
            stop_words='english')

        self.indexed = self.vectorizer.fit_transform(data)
        duration = time() - t0
        logger.debug("n_samples: %d, n_features: %d" % self.indexed.shape)
        logger.debug("duration: %d" % duration)
        return self.indexed

    def similarity(self, index=None, cap=0.5, k=100, batch_size=1000):
        '''
        Given a index (Matrix NxM) With N items and M features, calculates the similarity between each pair of items

        reference: https://stackoverflow.com/a/32885931/1354478
        :param index: Numpy matrix
        :return: Sparse matrix NxN where every cell is the similarity of its indexes
        '''
        if index is None:
            index = self.indexed
        super(CBAlgorithmJACCARD, self).similarity(index)

        # Get all the ids
        t0 = time()

        logger.debug(index.shape)
        matrix = index.todense()
        logger.debug('densed')

        # Start
        bar = ProgressBar(maxval=matrix.shape[0]/batch_size + 1, \
                      widgets=['JACCARD', ' ', Bar('=', '[', ']'), ' ', Percentage(), ' - ', Timer()])
        bar.start()
        # Calculate Similarity
        counter = 0
        for i in range(0, matrix.shape[0], batch_size):
            logger.debug("%d/%d", i, matrix.shape[0])
            m1 = matrix[i:i+batch_size,:]

            # Calculate Distance
            rows_movielens, cols_movielens, scores = distance_batch(m1, matrix, self.ids, self.ids, cap)

            # Extract TOP K result
            p = get_top_k(rows_movielens, cols_movielens, scores, k)

            # Temporarily save to a local file
            p.to_pickle('Temp/tmp_%s_%i' % (self.__name__, i))

            counter += 1
            bar.update(counter)
        bar.finish()

        # Append All Similarities
        frames = []
        for i in range(0, matrix.shape[0], batch_size):
            frames.append(pandas.read_pickle('Temp/%s_%i' % (db_fieldname, i)))
        result = pandas.concat(frames, axis=0)
        # result.to_pickle(db_fieldname)

        # Remove Temporary Files
        logger.debug('Duration: %f', time()-t0)
        return result
