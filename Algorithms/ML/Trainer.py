import itertools
import logging
import math
import numpy as np
import pandas
import Util.populate_similarity_table as populate_sim
from collections import OrderedDict
from DataHandler import PostgresDataHandler
from scipy import linalg
from sklearn import svm, linear_model, model_selection, metrics
from sklearn.preprocessing import StandardScaler
from Util import Field

logger = logging.getLogger('root')

class Trainer(object):
    def __init__(self, features, model=None):
        self.dataset = PostgresDataHandler()
        self.features = features
        if model is None:
            self.model = linear_model.Ridge(1.)
        else:
            self.model = model

    @staticmethod
    def get_model(name, *args):
        if name == 'svm':
            return svm.SVC(kernel='linear', C=.1)
            return svm.SVC(C=0.2)
        elif name == 'log_reg':
            return linear_model.LogisticRegression()
        else:
            return linear_model.Ridge(1.)

    @staticmethod
    def generate_features(filepath = None, fields = None, algorithms = None, k = None):
        '''
        Given the file path of a CSV with at least the following two columns:
            movieid1, movieid2,...
        Extract the unique movie ids and calculate the features to every possible combination of those ids

        :param filepath: Path to a csv file
        :return: None
        '''
        movies_ids_tagged = []
        if filepath is not None:
            # -*- Extract unique movie ids -*-

            # Load Data
            ground_truth = pandas.read_csv(filepath)

            # Delete empty records
            ground_truth.dropna()

            # Get unique ids
            m1 = ground_truth['movieid1'].unique()
            m2 = ground_truth['movieid2'].unique()
            movies_ids_tagged = np.union1d(m1, m2)
            logger.debug('No. of unique tagged IDs in ground_truth: %d', len(movies_ids_tagged))

        # Generate features
        populate_sim.main(movies_ids_tagged, fields, algorithms, k)
    
    @staticmethod
    def get_user_rating_als():
        '''
        Given the file path of a CSV with at least the following three columns:
            movieid1, movieid2, rating
        Return an aggregated pandas data frame with a single row for each movie pair, and its mean rating

        NOTE: If any of the movies in this file are not present in the TMDB database. Those rows will be removed

        :return: (user_ratings,deleted_register, clean_ground_truth)
        user_ratings: pandas data frame with the columns:
            movieid1: same as input
            movieid2: same as input
            rating: mean rating among all the users
        deleted_register: number of rows deleted because its movies are not in the TMDB Database
        clean_ground_truth: same dataframe as the input BUT removing the rows wich have movies
                            that are not present in the metadata table
        '''

        # Load Data
        dataset = PostgresDataHandler()

        ground_truth = pandas.DataFrame(dataset.get_als(), columns=['movieid1', 'movieid2', 'rating']);

        return Trainer.get_user_rating_from_df_movielens(ground_truth)

    @staticmethod
    def get_user_rating_uservote():
        '''
        Given the file path of a CSV with at least the following three columns:
            movieid1, movieid2, rating
        Return an aggregated pandas data frame with a single row for each movie pair, and its mean rating

        NOTE: If any of the movies in this file are not present in the TMDB database. Those rows will be removed

        :return: (user_ratings,deleted_register, clean_ground_truth)
        user_ratings: pandas data frame with the columns:
            movieid1: same as input
            movieid2: same as input
            rating: mean rating among all the users
        deleted_register: number of rows deleted because its movies are not in the TMDB Database
        clean_ground_truth: same dataframe as the input BUT removing the rows wich have movies
                            that are not present in the metadata table
        '''

        # Load Data
        dataset = PostgresDataHandler()

        ground_truth = pandas.DataFrame(dataset.get_uservote(), columns=['movieid1', 'movieid2', 'rating']);

        return Trainer.get_user_rating_from_df(ground_truth)

    @staticmethod
    def get_user_rating_from_df(dataframe):
        '''
        Given a non-aggregated pandas data frame with pairs of movie and boolean indicator of its rating
        Return an aggregated pandas data frame with a single row for each movie pair, and its mean rating

        NOTE: If any of the movies in this file are not present in the TMDB database. Those rows will be removed

        :param dataframe: Pandas dataframe with columns 'movieid1', 'movieid2', 'rating'
                          NOTE: The id of the movies should be their movielesns ID.
        :return: (user_ratings,deleted_register, clean_ground_truth)
        user_ratings: pandas data frame with the columns:
            movieid1: same as input
            movieid2: same as input
            rating: mean rating among all the users
        deleted_register: number of rows deleted because its movies are not in the TMDB Database
        clean_ground_truth: same dataframe as the input BUT removing the rows wich have movies
                            that are not present in the metadata table
        '''

        ground_truth = dataframe
        logger.debug('Original Record Numbers: %d', ground_truth.shape[0])

        # -*- Clean File using the MOVIE table as reference -*-

        # Extracts the unique movie dis
        m1 = ground_truth['movieid1'].unique()
        m2 = ground_truth['movieid2'].unique()
        ground_truth_ids = np.union1d(m1, m2)

        # Get movielens id stored in the DB
        dataset = PostgresDataHandler()
        db_ids = np.array(dataset.get_data(Field.MOVIELENS_ID))[:, 1].astype(int)

        # Extract ground_truth ids which are also in the DB
        valid_ground_truth_ids = np.intersect1d(db_ids, ground_truth_ids)

        # Get the ids which are NOT in the DB
        missing_movies = np.setdiff1d(ground_truth_ids, valid_ground_truth_ids)
        logger.debug('Missing Movies %d', len(missing_movies))

        # Remove any record that contains this movies
        mask1 = np.logical_not(ground_truth['movieid1'].isin(missing_movies))
        mask2 = np.logical_not(ground_truth['movieid2'].isin(missing_movies))
        clean_ground_truth = ground_truth.loc[mask1 & mask2]
        deleted_registers = ground_truth.shape[0] - clean_ground_truth.shape[0]
        logger.debug('Deleted Registers: %d' % (deleted_registers))

        # -*- Group By -*-
        # Group by MOVIEID1, MOVIEID2 using the MEAN as aggregator
        user_ratings = clean_ground_truth.groupby(['movieid1', 'movieid2'])['rating'].mean()
        user_ratings = user_ratings.reset_index()

        # Change ratings from number to binary
        user_ratings.loc['rating'] = user_ratings['rating'].values > 0.5

        logger.debug('Aggregated Record Numbers: %d', user_ratings.shape[0])

        return user_ratings, deleted_registers, clean_ground_truth

    @staticmethod
    def get_user_rating(filepath):
        '''
        Given the file path of a CSV with at least the following three columns:
            movieid1, movieid2, rating
        Return an aggregated pandas data frame with a single row for each movie pair, and its mean rating

        NOTE: If any of the movies in this file are not present in the TMDB database. Those rows will be removed

        :param filepath: Path to a csv file
        :return: (user_ratings,deleted_register)
        user_ratings: pandas data frame with the columns:
            movieid1: same as input
            movieid2: same as input
            rating: mean rating among all the users
        deleted_register: number of rows deleted because its movies are not in the TMDB Database
        '''

        # Load Data
        ground_truth = pandas.read_csv(filepath)
        logger.debug('Original Record Numbers: %d', ground_truth.shape[0])

        logger.debug(ground_truth.head())
        logger.debug(ground_truth.tail())
        # Delete empty records
        logger.debug("HEO")
        logger.debug(ground_truth.isnull().sum())
    
        ground_truth = ground_truth.dropna()
        logger.debug(ground_truth.isnull().sum())

        logger.debug(ground_truth[ground_truth.isnull().any(axis=1)])
    
        # -*- Clean File using the MOVIE table as reference -*-

        # Extracts the unique movie dis
        m1 = ground_truth['movieid1'].unique()
        m2 = ground_truth['movieid2'].unique()
        ground_truth_ids = np.union1d(m1, m2)

        # Get movielens id stored in the DB
        dataset = PostgresDataHandler()
        db_ids = np.array(dataset.get_data(Field.MOVIELENS_ID))[:, 1].astype(int)

        # Extract ground_truth ids which are also in the DB
        valid_ground_truth_ids = np.intersect1d(db_ids, ground_truth_ids)

        # Get the ids which are NOT in the DB
        missing_movies = np.setdiff1d(ground_truth_ids, valid_ground_truth_ids)
        logger.debug('Missing Movies %d', len(missing_movies))

        # Remove any record that contains this movies
        mask1 = np.logical_not(ground_truth['movieid1'].isin(missing_movies))
        mask2 = np.logical_not(ground_truth['movieid2'].isin(missing_movies))
        clean_ground_truth = ground_truth.loc[mask1 & mask2]
        deleted_registers = ground_truth.shape[0] - clean_ground_truth.shape[0]
        logger.debug('Deleted Registers: %d' % (deleted_registers))

        # -*- Group By -*-
        # Group by MOVIEID1, MOVIEID2 using the MEAN as aggregator
        user_ratings = clean_ground_truth.groupby(['movieid1', 'movieid2'])['rating'].mean()
        user_ratings = user_ratings.reset_index()

        # Change ratings from number to binary
        user_ratings['rating'] = user_ratings['rating'].values > 0.5

        logger.debug('Aggregated Record Numbers: %d', user_ratings.shape[0])

        return user_ratings, deleted_registers, clean_ground_truth

    @staticmethod
    def from_movielens_to_db_id(movielens_ids):
        """
        Given an array of movielens ids, return an array with the same dimensions with the Database ids
        :param movielens_ids: array of ints with movielens ids
        :return: array of strings with db ids
        """
        dataset = PostgresDataHandler()

        movie_ids = pandas.DataFrame(movielens_ids, columns=['movieid1'])
        # Get pairs (dbids - movielens_id)
        db_ids = np.array(dataset.get_data(Field.MOVIELENS_ID)).astype(int)  # TODO: Cache db_ids
        db_ids = pandas.DataFrame(db_ids, columns=['id1', 'movieid1'])

        # Merge movielens with dbids
        movie_ids = pandas.merge(movie_ids, db_ids, how="left", on='movieid1')
        movie_ids = movie_ids[['id1']]

        not_found = movie_ids[movie_ids.isnull().any(axis=1)]
        logger.debug('Records not found: %d', not_found.shape[0])

        return movie_ids.values.ravel().astype(int).tolist()

    # This one transform from movielens to db id
    def append_features(self, movie_pairs, standardized_flag):
        '''
        Given an array of movie pairs, it goes to the table mainsite_similarity and extract the specified features
        for each of the pairs. Finally it appends the features to the each movie pair
        :param movie_pairs: numpy matrix of 2 columns and N rows. Where the values are movielens movieid
        :return: A panda Data Frame with dimensions (2+K) x N. Where K is the number of features that the class was
                 initialized with and N is the original number of rows of the movie_pairs array.
        '''
        logger.debug('User Ratings: %d x %d', movie_pairs.shape[0], movie_pairs.shape[1])

        movie_pairs_db_id = np.zeros(shape=movie_pairs.shape).astype(int)
        movie_pairs_db_id[:, 0] = self.from_movielens_to_db_id(movie_pairs[:, 0])
        movie_pairs_db_id[:, 1] = self.from_movielens_to_db_id(movie_pairs[:, 1])
        movie_pairs_db_id = np.unique(movie_pairs_db_id, axis=0)
        features = self.dataset.get_features(movie_pairs_db_id.tolist(), self.features)

        features_df = pandas.DataFrame(features, columns=['movieid1', 'movieid2'] + self.features)
        logger.debug('User Ratings with Features: %d', features_df.shape[0])

        # -*- Standardize Coefficients -*-
        if standardized_flag:
            scaler = StandardScaler()
            scaler.fit(features_df.values[:,2:])
            features_df.iloc[:,2:] = scaler.transform(features_df.values[:,2:])
            
        movie_pairs_df = pandas.DataFrame(movie_pairs, columns=['movieid1', 'movieid2'])

        # If there is an error about empty or null values in the features
        # check if this merge is being done correctly
        # logger.debug(movie_pairs_df.shape)
        # logger.debug(features_df.shape)
        # tmp = pandas.merge(movie_pairs_df, features_df, how='inner')
        # logger.debug('User Ratings with Features (+): %dx%d', tmp.shape[0], tmp.shape[1])
        # tmp = pandas.merge(movie_pairs_df, features_df, how='left')
        # logger.debug('User Ratings with Features (left): %dx%d', tmp.shape[0], tmp.shape[1])
        
        result = pandas.merge(movie_pairs_df, features_df, how='left')

        return result

    # TODO: Find a more appropiate name for this function
    def append_features2(self, movie_pairs, standardized_flag):
        '''
        Given an array of movie pairs, it goes to the table mainsite_similarity and extract the specified features
        for each of the pairs. Finally it appends the features to the each movie pair
        :param movie_pairs: numpy matrix of 2 columns and N rows. Where the values are movielens movieid
        :return: A panda Data Frame with dimensions (2+K) x N. Where K is the number of features that the class was
                 initialized with and N is the original number of rows of the movie_pairs array.
        '''
        logger.debug('User Ratings: %d', len(movie_pairs))

        features = self.dataset.get_features(movie_pairs, self.features, flag_db_ids=True)
        logger.debug("features retrieved")

        features_df = pandas.DataFrame(features, columns=['movieid1', 'movieid2'] + self.features)
        logger.debug('User Ratings with Features: %d', features_df.shape[0])

        # -*- Standardize Coefficients -*-
        if standardized_flag:
            scaler = StandardScaler()
            scaler.fit(features_df.values[:,2:])
            features_df.iloc[:,2:] = scaler.transform(features_df.values[:,2:])
            
        movie_pairs_df = pandas.DataFrame(movie_pairs, columns=['movieid1', 'movieid2'])

        return pandas.merge(movie_pairs_df, features_df, how='left')

    def train(self, user_ratings_train, y_train, standardized_flag):
        '''
        Given an array of movies, and its respective booleans label trained the instance model
        :param user_ratings_train: numpy array with dimensions 2xN where the values are movielens ids
        :param y_train: array of len N, with boolean labels
        :return: a fitted model
        '''

        # Get Features
        x_train = self.append_features(user_ratings_train[:, [0, 1]], standardized_flag)
        logger.debug('Training with %d rows', x_train.shape[0])
        logger.debug('Training Data')
        logger.debug(x_train.head())

        # print(x_train[self.features].head())
        # # print(x_train[self.features].head())

        # Train
        self.model.fit(x_train[self.features], y_train)
        logger.debug('Model was trained')

        return self.model

    def predict(self, movielens_ids, user_ratings, k, standardized_flag):
        '''
        1- Given an array of movielens ids, and user ratings, look for each combination of the ids, with the ids
        in the user ratings.
        2- Predict a score with the previously fitted model.
        3- Retrieves only the top K elements for each movielen id
        :param movielens_ids: Int array of movilens ids
        :param user_ratings: Numpy array with the following columns (movieid1, movieid2, ...)
        :param k: Integer, with the wanted number of related movies
        :return: A panda data frame of dimensions (N * K, 2) where N is the size of the movilens id.
                 The first column will have the movielens id
                 The second column will have in descending order the best K movies for the id in the first column
        '''
        # Set the movielens id as the parent
        movie1_ids = movielens_ids

        # Get all the other movies, in this dataset evaluated by a user
        movie2_ids = user_ratings['movieid2'].unique()

        # Get all combinations of movie1-movie2
        lista = list(itertools.product(movie1_ids, movie2_ids))
        logger.debug('Predicting Combinations: %d', len(lista))

        return self.predict_from_pairs(np.array(lista), k, standardized_flag, remove=True)

    def predict_from_pairs(self, lista, k, standardized_flag, remove=False):
        '''
        1- Given an array of movielens ids, and user ratings, look for each combination of the ids, with the ids
        in the user ratings.
        2- Predict a score with the previously fitted model.
        3- Retrieves only the top K elements for each movielen id
        :param movielens_ids: Int array of movilens ids
        :param user_ratings: Numpy array with the following columns (movieid1, movieid2, ...)
        :param k: Integer, with the wanted number of related movies
        :return: A panda data frame of dimensions (N * K, 2) where N is the size of the movilens id.
                 The first column will have the movielens id
                 The second column will have in descending order the best K movies for the id in the first column
        '''
        
        # Look for the features for this pairs
        if remove:
            x_test = self.append_features(lista, standardized_flag)
        else:
            x_test = self.append_features2(lista, standardized_flag)

        logger.debug('Predict Features')
        logger.debug(x_test.head())

        # Predict score for all this features
        # Reference: http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
        coef = self.model.coef_.ravel() / linalg.norm(self.model.coef_)
        x_scores = np.dot(x_test[self.features], coef)


        x_scores = pandas.DataFrame(x_scores, columns=['Score'])
        logger.debug('Predicted')
        logger.debug(x_scores.head())

        # Concat ids and scores
        x_scores = pandas.concat([x_test[['movieid1', 'movieid2']], x_scores], axis=1)
        logger.debug('Predicted - concat')
        logger.debug(x_scores.head())

        # Get top K elements for each movieid1 set 1
        x_top = x_scores \
            .sort_values(by=['movieid1', 'Score'], ascending=False) \
            .groupby('movieid1') \
            .head(k) \
            .reset_index(drop=True)
        logger.debug('x_top')
        logger.debug(x_top.head())

        return x_top

    @staticmethod
    def get_metrics(true_labels, pred_labels, b_test=[]):
        '''
        Calculate the following metrics:
            - True Positive
            - True Negative
            - False Positive
            - False Negative
            - False Discovery Rate
            - Precision
            - Accuracy
            - Recall
            - F1
            - Average Precision
            - Mean Average Precision (MAP)
        :param true_labels: boolean array
        :param pred_labels: boolean array
        :param b_test: int id, which helps identify to which "block" or "movieid1" the results belong
        :return: A ordered dict with the previously described metrics
        '''
        # True Positive
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

        # True Negative
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

        # False Positive
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

        # False Negative
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

        logger.debug('TP: %i, FP: %i, TN: %i, FN: %i', TP, FP, TN, FN)

        fdr = FP / float(TP + FP)
        logger.debug('False Discovery Rate (FDR): %.5f', fdr)
        prec = metrics.precision_score(true_labels, pred_labels)
        logger.debug('Precission: %.5f', prec)
        acu = metrics.accuracy_score(true_labels, pred_labels)
        logger.debug('Accuracy: %.5f', acu)
        rec = metrics.recall_score(true_labels, pred_labels)
        logger.debug('Recall: %.5f', rec)
        f1 = metrics.f1_score(true_labels, pred_labels)
        logger.debug('F1: %.5f', f1)
        avg_prec = metrics.average_precision_score(true_labels, pred_labels)
        logger.debug('Average Precission: %.5f', avg_prec)

        average_precisions = []
        count = 0

        for i in set(b_test):
            if sum(true_labels[b_test == i]) == 0:
                # This happens because everything is false
                # in the groundtrugh, thus is impossible to calculate the MAP
                count += 1
                continue
            
            ap = metrics.average_precision_score(true_labels[b_test == i], pred_labels[b_test == i])
            average_precisions.append(ap)
        MAP = np.mean(average_precisions)
        logger.debug('Mean Average Precission: %.5f', MAP)
        logger.debug('Skipped records when calc. MAP %d', count)

        my_metrics = OrderedDict()
        my_metrics['TP'] = TP
        my_metrics['FP'] = FP
        my_metrics['TN'] = TN
        my_metrics['FN'] = FN
        my_metrics['MAP'] = MAP
        my_metrics['FDR'] = fdr
        my_metrics['Precision'] = prec
        my_metrics['Accuracy'] = acu
        my_metrics['Recall'] = rec
        my_metrics['F1'] = f1
        my_metrics['Average Precision'] = avg_prec

        return my_metrics

    @staticmethod
    def test(user_ratings_test, y_test, b_test, top_movie_pairs):
        '''
        Given a
        :param user_ratings_test: numpy array with the following columns (movieid1, movieid2, ...)
        :param y_test: boolean array with the labels for previous array
        :param b_test: int array with only the movieid1 #TODO: Check if we can remve this
        :param top_movie_pairs: numpy array of dimenxions Nx2 where each value is a movilens id
        :return: A ordered dictionary of metrics
        '''

        # Concat predictions with test set
        x_movie_pairs_test = pandas.DataFrame(user_ratings_test[:, [0, 1]].astype(int),
                                              columns=['movieid1', 'movieid2'])
        x_movie_pairs_test_scored = pandas.merge(x_movie_pairs_test, top_movie_pairs, how='left')

        # TODO: Remove this line
        test = pandas.merge(x_movie_pairs_test, top_movie_pairs, how='inner')
        if test.shape[0] == 0:
            logger.error('!!!Could not find any related movie!!!')
            logger.debug('test data %s', x_movie_pairs_test.dtypes)
            logger.debug(x_movie_pairs_test.head(20))
            logger.debug('predicted data %s', top_movie_pairs.dtypes)
            logger.debug(top_movie_pairs.head())
            logger.debug(top_movie_pairs[top_movie_pairs.movieid1 == 81834].values)
            return None

        y_predicted = x_movie_pairs_test_scored['Score'].notnull().values

        # Get metrics
        return Trainer.get_metrics(y_test > 0.5, y_predicted, b_test)

    # def evaluate(self, user_ratings, k = 20, standardized_flag=True):
    #     '''
    #     Given an aggregate panda dataframe of user ratings.
    #     - Split the dataset into TRAIN and TEST (50/50)
    #     - Train a model with the TRAIN part
    #     - Predict the top K movies with the TEST aprt
    #     - Test the predictions against the original user_ratings
    #     :param user_ratings: Pandas dataframe with the following columns (movieid1, movieid2, rating)
    #     :param k: number of related movies to pick for each id
    #     :return: a ordered dict of metrics
    #     '''

    #     # -*- Cross Validation Split -*-

    #     # Data
    #     logger.debug('EVALUATE')

    #     y = user_ratings['rating']

    #     logger.debug('\n\n\n\n\n\n\n\n\n--------------------------------------------------------')
    #     logger.debug(y.isnull().sum())
    #     logger.debug(y[y.isnull()])
        

    #     x = user_ratings.values
    #     blocks = user_ratings['movieid1'].values

    #     # split into train and test set
    #     cv = model_selection.GroupKFold(2)

    #     train_idx, test_idx = next(iter(cv.split(x, y, blocks)))
    #     user_ratings_train, y_train, b_train = x[train_idx], y[train_idx], blocks[train_idx]
    #     user_ratings_test, y_test, b_test = x[test_idx], y[test_idx], blocks[test_idx]

    #     # Train in FULL SET
    #     # X_train, y_train, b_train = X, y, blocks
    #     # X_test, y_test, b_test = X, y, blocks

    #     logger.debug('Total Data: %d', len(y))
    #     logger.debug('Intersection between Train/Test: %d', len(set(b_train).intersection(set(b_test))))
    #     logger.debug('Train Data Size: %d', len(user_ratings_train))
    #     logger.debug('Test Data Size: %d', len(user_ratings_test))

    #     # -*- Train -*-
    #     logger.info('Training')
    #     self.train(user_ratings_train, y_train, standardized_flag)

    #     # -*- Predict -*-
    #     logger.info('Predicting')
    #     top_movie_pairs = self.predict(set(b_test), user_ratings, k, standardized_flag)

    #     # -*- Test -*-
    #     logger.info('Testing')
    #     return self.test(user_ratings_test, y_test, b_test, top_movie_pairs)


    def evaluate(self, user_ratings, k = 20, standardized_flag=True):
        '''
        Given an aggregate panda dataframe of user ratings.
        - Split the dataset into TRAIN and TEST (50/50)
        - Train a model with the TRAIN part
        - Predict the top K movies with the TEST aprt
        - Test the predictions against the original user_ratings
        :param user_ratings: Pandas dataframe with the following columns (movieid1, movieid2, rating)
        :param k: number of related movies to pick for each id
        :return: a ordered dict of metrics
        '''

        # -*- Cross Validation Split -*-

        # Data
        y = user_ratings['rating'].values
        x = user_ratings.values
        blocks = user_ratings['movieid1'].values

        logger.debug('EVALUATE')

        # split into train and test set
        # TODO: Allow to tune the size of the training and testing set. Look into: model_selection.GroupShuffleSplit
        cv = model_selection.GroupKFold(2)

        train_idx, test_idx = next(iter(cv.split(x, y, blocks)))
        user_ratings_train, y_train, b_train = x[train_idx], y[train_idx], blocks[train_idx]
        user_ratings_test, y_test, b_test = x[test_idx], y[test_idx], blocks[test_idx]

        # Train in FULL SET
        # X_train, y_train, b_train = X, y, blocks
        # X_test, y_test, b_test = X, y, blocks

        logger.debug('Total Data: %d', len(y))
        logger.debug('Intersection between Train/Test: %d', len(set(b_train).intersection(set(b_test))))
        logger.debug('Train Data Size: %d', len(user_ratings_train))
        logger.debug('Test Data Size: %d', len(user_ratings_test))

        # -*- Train -*-
        logger.info('Training')
        logger.debug('\n\n\n\n\n\n\n\n\nB--------------------------------------------------------')
        logger.debug(np.isnan(y_train).sum())
        logger.debug(y_train[np.isnan(y_train)])
        self.train(user_ratings_train, y_train, standardized_flag)

        # -*- Predict -*-
        logger.info('Predicting')
        top_movie_pairs = self.predict(set(b_test), user_ratings, k, standardized_flag)

        # -*- Test -*-
        logger.info('Testing')
        return self.test(user_ratings_test, y_test, b_test, top_movie_pairs)

    def train_with_full_dataset(self, user_ratings, features_field, k=20, standardized_flag=False):

        # Data
        y = user_ratings['rating']
        x = user_ratings.values
        
        # -*- Train -*-
        self.train(x, y, standardized_flag)

        return self.model

        