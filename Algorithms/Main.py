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

class Dataset(object):

    def get_data(self, field):
        return ['this is a practice dataset',
                'only used to test functionality',
                'hola',
                'hola este es mi mejor momento',
                'quizas no']


from CB import CBRecommender, CBAlgorithmTFIDF, CBAlgorithmBM25, CBAlgorithmJACCARD

def main(argv):
    logging.basicConfig(level=logging.WARN)

    # Get the recommender
    rec = CBRecommender()

    # Get the connection with the 'database'
    dataset = Dataset()

    for algo in [CBAlgorithmTFIDF(), CBAlgorithmBM25(), CBAlgorithmJACCARD()]:
        # Train the recommender using the given algorithm and dataset
        result = rec.train('title', algo, dataset)

        # Report similarity score for every single pair that has a score > 0
        logging.warning('\n%s\n%s' % (algo.__name__, str(result)))


if __name__ == "__main__":
    main(sys.argv)