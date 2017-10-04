# -*- coding: utf-8 -*-
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

import logging

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

import sys
import logging
import numpy as np
from DataHandler import PostgresDataHandler
from Util import Field
from ML.Trainer import Trainer
from progressbar import ProgressBar, Bar, Percentage, Timer


logger.info('Start')

def main(argv):
    features_field = ['title_tfitf',
        'title_bm25',
        'title_jaccard',
        'genre_tfitf',
        'genre_bm25',
        'genre_jaccard',
        'director_tfitf',
        'director_bm25',
        'director_jaccard',
        'writer_tfitf',
        'writer_bm25',
        'writer_jaccard',
        'cast_tfitf',
        'cast_bm25',
        'cast_jaccard',
        'plot_tfitf',
        'plot_bm25',
        'plot_jaccard',
        'full_plot_tfitf',
        'full_plot_bm25',
        'full_plot_jaccard',
        'language_tfitf',
        'language_bm25',
        'language_jaccard',
        'country_tfitf',
        'country_bm25',
        'country_jaccard',
        'awards_tfitf',
        'awards_bm25',
        'awards_jaccard']

    bar = ProgressBar(widgets=[Timer()]).start()
    trainer = Trainer(features_field)

    # WARNING: This takes 10 minutes
    # trainer.generate_features(r'./Data/groundtruth.exp1.csv')

    user_ratings, deleted_registers  = trainer.get_user_rating(r'./Data/groundtruth.exp1.csv')
    result = trainer.evaluate(user_ratings)
    print '\n -*- Metrics -*-'
    for key in result:
        print '%s %f' % (key.ljust(10), result[key])
    bar.finish()

if __name__ == "__main__":
    main(sys.argv)