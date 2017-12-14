# -*- coding: utf-8 -*-
"""
               Recommender Engine
===================================================

This module implements recommendations algorithms:
    - CB (Content Based)
        - TFIDF
        - BM25
        - JACCARD
    - CF (Collaborative Filtering)
        - Latent Factors -> ALS

It allows persisting features and recommendations using DataHandlers:
    - PostgreSQL

Finally, it combines the previously generated features using one of the machine learning models:
    - Linear Regression
    - Logistic Regression
    - Support Vector Machine

===================================================
Usage: Main.py [-h] [-algorithm [Algorithm ID]] [-movie [Movie ID]] command [file_path]

Commands:
    - test:     Use a ground truth file to evaluate effectiveness of some feature combination
                Set the features in the settings.py file
    - train:    Use a complete ground truth file to train a machine learning model and persist it to the file system
    - als:      Train a Matrix Factorization Model and generate cosine similarity features between the movies vectors
                of the train model, using the library MyMediaLite
    - als_libmf Train a Matrix Factorization Model and generate cosine similarity features between the movies vectors
                of the train model, using the library LIBMF.
    - populate  Calculate and persist the best 20 movie recommendations for each of the movies in the database
    - predict   Given a movie id it predicts the best 20 movie recommendations for that movie
    - add_tmdb  Aggregates the recommendations of the tmdb website as an extra algorithm to the database.
    - age_diff  Populate the age_diff feature in the features table
    - add_field Create the temporary table and a field for each algorithm in the features table

"""

# Author: Caleb De La Cruz P. <delacruzp>

import argparse
import logging
import os
import numpy as np
import settings as s
from DataHandler import PostgresDataHandler
from ML.Trainer import Trainer
from progressbar import ProgressBar, Timer
from sklearn.externals import joblib
from Util.populate_similarityals import als
from Util.populate_similaritylibmf import libmf_als
from Util.add_tmdb_movie_similarity import add_tmdb
from Util.field_add import add
from Util.enums import Field

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)3s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)


def test(file_path):
    """
    Input:
    - Ground truth file path
    
    Output:
    - None

    Standard Output:
        - Metrics (TF, Precision, Recall, etc.)
    

    Test the performance of features in the 'features_field' against
    a provided ground truth file.

    Modify features field to try out different algorithms or 
    combinations of fields.

    Disclaimer: By default all the features are always generated for
    all the movies, this allows to run the features generation just
    once at the beginning an then play with the features. If you 
    don't want for any reason to do this, you can specify which
    algorithms you want to generate features for in the 
    "trainer.generate_features" function

    Detail:
    Given a the path of a ground truth file
        1- Takes the movielens ids of the movies in the file
        2- Generate features for each pair of the movies in that file
        3- Calculate the score of each  pair using the user's label
        4- Train a Linear Regression using the fields in the 
           global variable features_field as features and
           using the 50% of the data in the file provided labels
        5- Predict similar movies for the remaining 50% of the data
           in the file provided
        6- Measure the performance of the predictions against
           what the users considered similar or not similar according
           to the file provided
    """
    bar = ProgressBar(widgets=[Timer()]).start()

    model = Trainer.get_model(s.model)
    trainer = Trainer(s.features_field, model)

    if s.generate_features:
        # TODO: Add all this params to settings.py
        # filepath = None, fields = None, algorithms = None, k = None
        trainer.generate_features(fields=[Field.TAGS])
        # trainer.generate_features(fields=[Field.FULL_PLOT, Field.TAGS])

    user_ratings, deleted_registers, full_user_ratings = trainer.get_user_rating(file_path)

    if s.aggregated_training_data:
        # Aggregated Version
        result = trainer.evaluate(user_ratings, s.k, s.standardized_coefficients)
    else:
        # Non-Aggregate Version
        result = trainer.evaluate(full_user_ratings, s.k, s.standardized_coefficients)

    print('\n -*- Features -*-')
    for i in sorted(zip(s.features_field, trainer.model.coef_.ravel()), key=lambda x: abs(x[1])):
        print(i)

    print('\n -*- Metrics -*-')
    for key in result:
        print('%s\t%f' % (key.ljust(10), result[key]))
    
    print('\n -*- Metrics [Only Values] -*-')
    for key in result:
        print('%f' % (result[key]))

    bar.finish()


def train(file_path):
    """
    Use the 100% of the data in a given ground truth file
    to train the linear regression model. Finally it serializes
    and save the model in the file system as (TMP_MODEL.pkl) for
    future use.

    Input:
    - Ground truth file path
    - Optional: Linear Regression model file path

    Output:
    - Model

    Standard Output:
    - Weight of each feature
    """

    bar = ProgressBar(widgets=[Timer()]).start()

    model = Trainer.get_model(s.model)
    trainer = Trainer(s.features_field, model)

    user_ratings, deleted_registers, full_user_ratings = trainer.get_user_rating(file_path)

    if s.aggregated_training_data:
        # Aggregated Version
        model = trainer.train_with_full_dataset(user_ratings, s.features_field, s.k, s.standardized_coefficients)
    else:
        # Non-Aggregate Version
        model = trainer.train_with_full_dataset(full_user_ratings, s.features_field, s.k,
                                                s.standardized_coefficients)

    print('\n -*- Features -*-')
    for i in sorted(zip(s.features_field, model.coef_), key=lambda x: x[1]):
        print(i)

    print('\n -*- Persisting Model -*-')
    joblib.dump(model, s.model_file_path)
    
    print('\n -*- Model succesfully trained -*-')
    bar.finish()
    return model


def predict_movie(trainer,
                  low,
                  high=None,
                  algorithm=None):
    """
    Given a trained model it would generate movie recommendations for any movie in the range [low,high] inclusive.

    If an algorithm id is given, the recommendations will be persisted to the database with that id, as identifier.

    :param trainer: Trained machine learning model
    :param low: Movie Id, that identifies the start of the range
    :param high: Movie Id, that identifies the end of the range
    :param algorithm: Optional algorithm id, which is used to decide if persist or not the recommendations
    :return: Movie recommendation for the range
    """

    if high is None:
        high = low + 1

    # Get all the pairs saved into the main site_similarity DB
    # using movieid 1 between a range and all the movieid 2 saved
    features = trainer.dataset.get_pairs(low=low, high=high)

    features = np.array(features)
    print('\tPairs being predicted: ', len(features))

    # -*- Predict -*-
    print('-*- predicting -*-')
    top_movie_pairs = trainer.predict_from_pairs(features, s.k, s.standardized_coefficients)
    print('Predicted pairs: ', len(top_movie_pairs))
    print('Predicted pairs: ', top_movie_pairs)
    # -*- Persist -*-
    if algorithm is not None:
        trainer.datas.clear_similar_movies(algorithm)
        trainer.datas.save_similar_movies(top_movie_pairs.values.tolist(), algorithm)
        print('-*- similar movies succesfully save -*-')

    return top_movie_pairs


def predict(movie_id, file_path):
    """
    Give a movie id, trains a model and generate movie recommendations for that model
    :param movie_id: Movie ID
    :param file_path: Ground truth File Path
    """
    model = train(file_path)
    print('-*- model loaded -*-')
    trainer = Trainer(s.features_field, model)

    predict_movie(trainer, movie_id, algorithm=s.discard_algorithm_id)


def populate_movie_pairs(algorithm, file_path=None):
    """
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
    """

    # Checking file_path is valid
    model = None
    if os.path.isfile(s.model_file_path):
        model = joblib.load(s.model_file_path)
    else:
        if file_path is None:
            logger.warning("A model should be trained before populating the database.")
            exit(-1)
        else:
            model = train(file_path)

    print('-*- model loaded -*-')
    trainer = Trainer(s.features_field, model)

    trainer.dataset.clear_similar_movies(algorithm)
    for i in range(s.minimum, s.maximum, s.steps):
        print('Update: ', i, '/', s.maximum)

        # Get all the pairs saved into the main site_similarity DB
        # using movieid1 between a range and all the movieid2 saved
        features = trainer.dataset.get_pairs(low=i, high=i+s.steps)

        features = np.array(features)
        print('\tPairs being predicted: ', len(features))

        # -*- Predict -*-
        print('-*- predicting -*-')
        top_movie_pairs = trainer.predict_from_pairs(features, s.k, s.standardized_coefficients)
        print('Predicted pairs: ', len(top_movie_pairs))

        # -*- Persist -*-
        trainer.dataset.save_similar_movies(top_movie_pairs.values.tolist(), algorithm)
        print('-*- similar movies successfully save -*-')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recommendation Engine.')
    parser.add_argument('command', metavar='command', help='Sub-command',
                        choices=['test', 'train', 'als', 'als_libmf', 'age_diff', 
                        'add_tmdb', 'populate', 'predict', 'add_field'])
    parser.add_argument('file_path', metavar='file_path', nargs='?', default=r'./Data/groundtruth.exp1.csv',
                        help='Groundtruth File')
    parser.add_argument('-algorithm', metavar='Algorithm ID', type=int, nargs='?', default=-1,
                        help='An integer to group all the recommendations generated by the populate function')
    parser.add_argument('-movie', metavar='Movie ID', type=int, nargs='?', default=-1,
                        help='The movielens ID')
    parser.add_argument('-fieldname', metavar='Field Name', nargs='?',
                        help='Name of the new field being added')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    command = args.command

    # Adjusting Verbosity
    if args.verbose < 1:
        logger.setLevel(logging.WARN)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    # Checking file_path is valid
    if not os.path.isfile(args.file_path):
        print("Invalid Filepath: %s" % args.file_path)
        exit(1)

    if command == 'test':
        test(args.file_path)
    elif command == 'train':
        train(args.file_path)
    elif command == 'als':
        als()
    elif command == 'als_libmf':
        libmf_als()
    elif command == 'age_diff':
        PostgresDataHandler().updateSimilarityAgeDiffFeature()
    elif command == 'add_tmdb':
        add_tmdb()
    elif command == 'populate':
        populate_movie_pairs(args.algorithm, args.file_path)
    elif command == 'predict':
        predict(args.movie, args.file_path)
    elif command == 'add_field':
        add(args.fieldname)
