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
from Util.populate_similaritylibmf import libmf_als, recreate_Similarity_Table
from Util.add_tmdb_movie_similarity import add_tmdb

features_field = [
        # 'title_tfitf',
        # 'genre_tfitf',
        # 'director_tfitf',
        # 'writer_tfitf',
        # 'cast_tfitf',
        # 'plot_tfitf',
        # 'full_plot_tfitf',
        # 'language_tfitf',
        # 'country_tfitf',
        # 'awards_tfitf',
        'title_bm25',
        'genre_bm25',
        'director_bm25',
        'writer_bm25',
        'cast_bm25',
        'plot_bm25',
        'full_plot_bm25',
        'language_bm25',
        'country_bm25',
        'awards_bm25',
#         'title_jaccard',
#         'genre_jaccard',
#         'director_jaccard',
#         'writer_jaccard',
#         'cast_jaccard',
#         'plot_jaccard',
#         'full_plot_jaccard',
#         'language_jaccard',
#         'country_jaccard',
#         'awards_jaccard'
        'year',
#         'released',
#         'metacritic',
#         'imdb_rating',
#         'imdb_votes',
#         'popularity',
#         'budget',
        'als_cosine',
#         'weighted_als_cosine',
]


def test(filepath):
    '''
    Input:
    - Groundtruth file path
    
    Output:
    - None

    Standard Output:
    - Metrics (TF, Precission, Recall, etc.)
    

    Test the performance of features in the 'features_field' against
    a provided groundtruth file. 

    Modify features field to try out different algorithms or 
    combinations of fields.

    Disclaimer: By default all the features are always generated for
    all the movies, this allows to run the features generation just
    once at the beginning an then play with the features. If you 
    don't want for any reason to do this, you can specify which
    algorithms you want to generate features for in the 
    "trainer.generate_features" function

    Detail:
    Given a the path of a groundtruth file
        1- Takes the movielens ids of the movies in the file
        2- Generate features for each pair of the movies in that file
        3- Calculate the score of each  pair using the user's label
        4- Train a Linear Regression using the fields in the 
           global variable features_field as features and
           using the 50% of the data in the file provided labels
        5- Predict similar movies for the reamaining 50% of the data 
           in the file provided
        6- Measure the performance of the predictions against
           what the users considered similar or not similar according
           to the file provided
    '''
    bar = ProgressBar(widgets=[Timer()]).start()
    trainer = Trainer(features_field)

    # WARNING: This takes 10 minutes
    # trainer.generate_features(fields=[Field.FULL_PLOT])

    user_ratings, deleted_registers, full_user_ratings  = trainer.get_user_rating(filepath)
    result = trainer.evaluate(full_user_ratings)
    # result = trainer.evaluate(user_ratings)
    
    print('\n -*- Features -*-')
    t = list(zip(features_field, trainer.model.coef_))
    for i in sorted(zip(features_field, trainer.model.coef_), key=lambda x: abs(x[1])):
        print(i)

    print('\n -*- Metrics -*-')
    for key in result:
        print('%s %f' % (key.ljust(10), result[key]))
    
    for key in result:
        print('%f' % (result[key]))
    
    bar.finish()

    predict_movie(trainer, 25175, algorithm=-1)

def predict(id):
    model = train(r'./Data/groundtruth.exp1.csv')
    print('-*- model loaded -*-')
    trainer = Trainer(features_field, model)

    predict_movie(trainer, id, algorithm=-1)

def train(filepath, model_filepath='TMP_MODEL.pkl'):
    '''
    Use the 100% of the data in a given groundtruth file
    to train the linear regression model. Finally it serializes
    and save the model in the file system as (TMP_MODEL.pkl) for
    future use.

    Input:
    - Groundtruth file path
    - Optional: Linear Regression model file path

    Output:
    - Model

    Standard Output:
    - Weight of each feature
    '''

    bar = ProgressBar(widgets=[Timer()]).start()
    trainer = Trainer(features_field)
    user_ratings, deleted_registers, full_dataset = trainer.get_user_rating(filepath)
    model = trainer.train_with_full_dataset(full_dataset, features_field)
    print('\n -*- Features -*-')
    for i in sorted(zip(features_field, model.coef_), key=lambda x: x[1]):
        print(i)

    print('\n -*- Persisting Model -*-')
    joblib.dump(model, model_filepath) 
    
    print('\n -*- Model succesfully trained -*-')
    bar.finish()
    return model

def predict_movie(trainer, low, high=None, standardized_flag=False, k=20, algorithm=None):
    if high is None:
        high = low + 1

    # Get all the pairs saved into the mainsite_similarity DB
    # using movieid1 between a range and all the movieid2 saved
    features = trainer.dataset.get_pairs(low=low, high=high)

    features = np.array(features)
    print('\tPairs being predicted: ', len(features))

    # -*- Predict -*-
    print('-*- predicting -*-')
    top_movie_pairs = trainer.predict_from_pairs(features, k, standardized_flag)
    print('Predicted pairs: ', len(top_movie_pairs))
    print('Predicted pairs: ', top_movie_pairs)
    # -*- Persist -*-
    if algorithm is not None:
        trainer.dataset.clear_similar_movies(algorithm)
        trainer.dataset.save_similar_movies(top_movie_pairs.values.tolist(), algorithm)
        print('-*- similar movies succesfully save -*-')

    return top_movie_pairs
    
def populate_movie_pairs(model_filepath='TMP_MODEL.pkl', algorithm=0):
    '''
    Use a previously trained model:
        You NEED to first run the train function to generate that file.

    Then it will try to predict the best 20 similar movies, between
    each possibly movie pair.

    Input:
    - Linear Regression model file path

    Output:
    - Insert into Movie Pair Table

    Standard Output:
    - n/a
    '''

    # model = joblib.load(model_filepath) 
    model = train(r'./Data/groundtruth.exp1.csv')
    print('-*- model loaded -*-')
    trainer = Trainer(features_field, model)

    minimum = 3
    maximum = 27001

    step = 2000
    trainer.dataset.clear_similar_movies(algorithm)
    for i in range(0,27300,step):
        print('Update: ', i, '/', 27300)

        # Get all the pairs saved into the mainsite_similarity DB
        # using movieid1 between a range and all the movieid2 saved
        features = trainer.dataset.get_pairs(low=i, high=i+step)

        features = np.array(features)
        print('\tPairs being predicted: ', len(features))

        # -*- Predict -*-
        standardized_flag = False
        k = 20

        print('-*- predicting -*-')
        top_movie_pairs = trainer.predict_from_pairs(features, k, standardized_flag)
        print('Predicted pairs: ', len(top_movie_pairs))
        # -*- Persist -*-
        trainer.dataset.save_similar_movies(top_movie_pairs.values.tolist(), algorithm)
        print('-*- similar movies succesfully save -*-')
        
# from CB.CBAlgorithmTMDB import CBAlgorithmTMDB
# data = list(CBAlgorithmTMDB().ranking())
# print(data[:10])
# m = []
# for i in data:
#     for j in i[1]:
#         m.append([i[0],j])
# m = np.array(m)
# m[:,0] = Trainer.from_movielens_to_db_id(m[:,0])
# m[:,1] = Trainer.from_movielens_to_db_id(m[:,1])
# m = m[~(m[:,1] == -2147483648)]
# m = m[~(m[:,0] == -2147483648)]
# # m = m[~np.isnan(m).any(axis=1)]

# PostgresDataHandler().save_similar_movies(m, 3)
# print('-*- similar movies succesfully save -*-')
        
# filepath = r'./Data/groundtruth.exp1.csv'

# features_field = [
#         'title_bm25',
#         'genre_bm25',
#         'director_bm25',
#         'writer_bm25',
#         'cast_bm25',
#         'plot_bm25',
#         'full_plot_bm25',
#         'language_bm25',
#         'country_bm25',
#         'awards_bm25',
#         'year',
# ]

# populate_movie_pairs(filepath, 4)

# features_field = [
#         'title_tfitf',
#         'genre_tfitf',
#         'director_tfitf',
#         'writer_tfitf',
#         'cast_tfitf',
#         'plot_tfitf',
#         'full_plot_tfitf',
#         'language_tfitf',
#         'country_tfitf',
#         'awards_tfitf',
#         'year',
# ]

# populate_movie_pairs(filepath, 0)


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
    elif command == 'populate_age_diff':
        dataset = PostgresDataHandler()
        dataset.updateSimilarityAgeDiffFeature();
    elif command == 'add_tmdb':
        add_tmdb()
    elif command == 'p' or command == 'populate':
        populate_movie_pairs(filepath, -1)
    elif command == 'predict':
        predict(int(filepath))
    elif command == 'recreate_similarity':
        recreate_Similarity_Table()
    else:
        print('\nUsage: python main.py <command> [filepath] \
        \n \
        \nInvalid command: \'' + command + '\' valid choices are: \
        \n \
        \n test \
        \n train \
        \n p \
        \n populate ')
