"""
               Recomender Engine
===================================================

This module implements recommendations algorithms:
    - CB (Content Based)
        - TFIDF
        - BM25
        - JACCARD

"""

# Author: Caleb De La Cruz P. <cdelacru>

import sys
import logging

#~~~                                    Do a basic silly run                                 ~~~#
from DataHandler import PostgresDataHandler
from Util import Field




from CB import CBRecommender, CBAlgorithmTFIDF, CBAlgorithmBM25, CBAlgorithmJACCARD

def main(argv):
    logging.basicConfig(level=logging.WARN)

    # Get the recommender
    rec = CBRecommender()

    # Get the connection with the 'database'
    dataset = PostgresDataHandler()


    # Single Field
    data = dataset.get_data(Field.CAST)

    logging.warn("%d records retrieved", len(data))
    for algo in [CBAlgorithmTFIDF(), CBAlgorithmBM25(), CBAlgorithmJACCARD()]:
        # Train the recommender using the given algorithm and dataset
        result = rec.train(data, algo)

        # Report similarity score for every single pair that has a score > 0
        logging.warning('\nAlgorithm: %s\nComparisons: %s\nExample Data%s' % (algo.__name__, format(result.nnz, ",d"), str(result[0,0:10])))

    # Multiple Field

    logging.warn("%d records retrieved", len(data))
    for algo in [CBAlgorithmTFIDF(), CBAlgorithmBM25(), CBAlgorithmJACCARD()]:
        # Train the recommender using the given algorithm and dataset
        result = rec.train_several_fields([Field.CAST, Field.TITLE, Field.PLOT], algo, dataset, [0.3, 0.3, 0.4])

        # Report similarity score for every single pair that has a score > 0
        logging.warning('\nAlgorithm: %s\nComparisons: %s\nExample Data%s' % (algo.__name__, format(result.nnz, ",d"), str(result[0,0:10])))


if __name__ == "__main__":
    main(sys.argv)