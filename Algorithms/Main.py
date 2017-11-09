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
FORMAT = "[%(filename)s:%(lineno)3s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

import sys
import logging
import numpy as np
from DataHandler import PostgresDataHandler
from Util import Field
from ML.Trainer import Trainer
from progressbar import ProgressBar, Bar, Percentage, Timer
from sklearn.externals import joblib
from Util.populate_similarityals import als
from Util.populate_similaritylibmf import libmf_als
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
        'awards_jaccard',
        'filtered_plot_tfitf',
        'filtered_plot_bm25',
        'filtered_plot_jaccard',
        'als_cosine']


def test(filepath):
    bar = ProgressBar(widgets=[Timer()]).start()
    trainer = Trainer(features_field)

    #WARNING: This takes 10 minutes
    #trainer.generate_features(r'./Data/groundtruth.exp1.csv')

    user_ratings, deleted_registers  = trainer.get_user_rating(filepath)
    result = trainer.evaluate(user_ratings)
    print('\n -*- Metrics -*-')
    for key in result:
        print('%s %f' % (key.ljust(10), result[key]))
    bar.finish()

def train(filepath):
    bar = ProgressBar(widgets=[Timer()]).start()
    trainer = Trainer(features_field)
    user_ratings, deleted_registers  = trainer.get_user_rating(filepath)
    model = trainer.train_with_full_dataset(user_ratings, features_field)
    print('\n -*- Features -*-')
    for i in sorted(zip(features_field, model.coef_), key=lambda x: x[1]):
        print(i)

    print('\n -*- Persisting Model -*-')
    joblib.dump(model, 'TMP_MODEL.pkl') 
    
    print('\n -*- Model succesfully trained -*-')
    bar.finish()

def populate_movie_pairs(filepath):

    model = joblib.load('TMP_MODEL.pkl') 
    print('-*- model loaded -*-')
    trainer = Trainer(features_field, model)

    minimum = 3
    maximum = 27001

    step = 300
    trainer.dataset.clear_similar_movies()
    for i in range(0,27300,step):
        print('Update: ', i, '/', 27300)

        features = trainer.dataset.get_pairs(low=i, high=i+step)

        features = np.array(features)
        print('Pairs being predicted: ', len(features))

        # -*- Predict -*-
        standardized_flag = False
        k = 20

        print('-*- predicting -*-')
        top_movie_pairs = trainer.predict_from_pairs(features, k, standardized_flag)
        print('Predicted pairs: ', len(top_movie_pairs))
        # -*- Persist -*-
        trainer.dataset.save_similar_movies(top_movie_pairs.values.tolist())
        print('-*- similar movies succesfully save -*-')
        

if __name__ == "__main__":

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = 'test'
    
    if len(sys.argv) > 2:
        filepath = sys.argv[2]
    else:
        filepath = r'./Data/groundtruth.exp1.csv'
    
    if command == 'test':
        test(filepath)
    elif command == 'train':
        train(filepath)
    elif command == 'als':
        als()
    elif command == 'als-libmf':
        libmf_als()
    elif command == 'p' or command == 'populate':
        populate_movie_pairs(filepath)
    else:
        print('\nUsage: python main.py <command> [filepath] \
        \n \
        \nInvalid command: \'' + command + '\' valid choices are: \
        \n \
        \n test \
        \n train \
        \n p \
        \n populate ')
