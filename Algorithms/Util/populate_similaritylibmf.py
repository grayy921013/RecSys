import logging

import pandas
import os
import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.sparse import csr_matrix
from numpy import genfromtxt
from progressbar import ProgressBar, Bar, Percentage, Timer
from DataHandler.Postgres import PostgresDataHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def libmf_als(batch_size=100, cap=0.5, k=100):
    db_fieldname = 'libmf_cosine'
    #generate_libmf_als()
    libmf_transform()
    matrix = genfromtxt('Temp/model_movies1.csv', delimiter=',')
    data = cosine_similarity(matrix, batch_size, cap, k)
    data = pandas.read_pickle('Temp/'+db_fieldname)
    dataset = PostgresDataHandler()
    dataset.save_libmf(data[['id1_id', 'id2_id', 'libmf_cosine']])

def recreate_similarity_table():
    dataset = PostgresDataHandler()
    dataset.clear_similarity()


def generate_libmf_als():
    # command = './rating_prediction --training-file=ratings.csv --recommender=MatrixFactorization --test-ratio=0.1  --save-user-mapping=user_mapping.txt --save-item-mapping=item_mapping.txt  --save-model=model.txt'
    output = os.system('pwd')
    print('pwd', output)

    if os.name == 'nt':
        raise Exception('There is no a LibMF executable for windows')
    else:
        executable = './CF/libMF/mf-train'
        #executable = './mf-train'

    command = '''%s -f 0 -l2 0.05 -k 100 -t 10 ./Data/ml-20m/ratings_libmf_tr.txt model1.txt''' % executable
    output = os.system(command)
    print(output)


def cosine_similarity(matrix, batch_size, cap, k):
    m2 = matrix
    db_fieldname = 'libmf_cosine'

    bar = ProgressBar(maxval=matrix.shape[0] / batch_size + 1, \
                      widgets=['ALS: Cosine', ' ', Bar('=', '[', ']'), ' ', Percentage(), ' - ', Timer()])
    bar.start()

    print("Calculate Similarities")
    # Calculate Similarity
    counter = 0
    for i in range(0, matrix.shape[0], batch_size):
        m1 = matrix[i:i + batch_size, :]
        cosine_similarity_batch(m1, m2, cap, k, '%s_%i' % (db_fieldname, i))
        counter += 1
        bar.update(counter)
    bar.finish()
    print("Append All Similarities")
    # Append All Similarities
    frames = []
    for i in range(0, matrix.shape[0], batch_size):
        frames.append(pandas.read_pickle('Temp/%s_%i' % (db_fieldname, i)))
    result = pandas.concat(frames, axis=0)
    print("result.to_pickle(db_fieldname)")
    result.to_pickle('Temp/'+db_fieldname)
    return result


def cosine_similarity_batch(m1, m2, cap, k, db_fieldname):
    # Get all the ids
    ids_1 = m1[:, 0].astype(int)
    ids_2 = m2[:, 0].astype(int)

    m1 = m1[:, 1:]
    m2 = m2[:, 1:]

    result = 1 - cdist(m1, m2, 'cosine')
    result = np.array(result)
    result[result < cap] = 0
    # print 'result size', len(result)
    # print 'result size', np.array(result).shape

    # mask_too_far = result < cap
    # result = result[mask_too_far]
    # print 'result prune', len(result)

    result = csr_matrix(result)
    # print 'result sparse matrix', result


    # from itertools import product
    # comb = np.array(list(product(ids_1, ids_2)))
    # print 'comb.shape', comb.shape
    # # mask = comb[:,0] > comb[:,1]
    # # comb = comb[mask]
    # # print 'comb.shape', comb.shape

    # comb = comb[mask_too_far]
    # print 'comb.shape prune', comb.shape

    # Extract the data from the sparse matrix
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
    print("Before getting top k")
    p = get_top_k(rows_movielens, cols_movielens, scores, k)

    # Temporarily save to a local file
    print("p.to_pickle(db_fieldname)")
    p.to_pickle('Temp/'+db_fieldname)


def get_top_k(rows_movielens, cols_movielens, scores, k):
    # Only save the ids of the movies for the first
    # algorithm, because next one will have the same id
    pre_frame = np.rec.fromarrays((rows_movielens, cols_movielens, scores), \
                                  names=('id1_id', 'id2_id', 'libmf_cosine'))

    p = pandas.DataFrame(pre_frame)
    print("Getting top k elements for each movieid")
    # Get top K elements for each movieid1 set 1

    p = p \
        .sort_values(by=['id1_id', 'libmf_cosine'], ascending=False) \
        .groupby('id1_id') \
        .head(k) \
        .reset_index(drop=True)
    print("Sort")
    p = p.sort_values(by=['id1_id'], ascending=True)
    print("Temporarily save to a local file")
    # Temporarily save to a local file
    logger.info('ALS\t %d/%d found/saved', p.shape[0], len(scores))

    return p


def libmf_transform():
    f = open('model1.txt')
    f_movies = open('Temp/model_movies1.csv', 'w')




    # Skip top of the model and user matrix
    line = '123'
    movies = 0
    features = 0
    line = f.readline()
    while line[0] != 'n':
        print(line + " line ")
        line = f.readline()

    parts = line.split()
    movies = int(parts[1])
    line = f.readline()
    parts = line.split()
    features = int(parts[1])
    print(movies)
    print(features)
    #movies, features = list(map(int, f.readline().split()))
    #q = false
    while line[0] != 'q':
        line = f.readline()

    # Read # of movies and # of features
    #movies, features = list(map(int, f.readline().split()))
    for i in range(movies -1):

        m_features = [None] * features
        m_id = 0
        counter = 0

        lineparts = f.readline().split()

        m_id = lineparts[0][1:]
        idx = 0
        for itr in range(features):
            m_value = lineparts[2 + idx]
            m_features[itr] = m_value
            counter += float(m_value)
            idx = idx + 1


        if counter:
            m_features = [m_id] + m_features
            f_movies.write(','.join(m_features) + '\n')
    f.close()
    f_movies.close()

# def transform():
#     f = open('model.txt')
#     f_movies = open('Temp/model_movies.csv','w')
#
#     # Skip top of the model and user matrix
#     line = '123'
#     while len(line) > 1:
#         line = f.readline()
#
#     # Read # of movies and # of features
#     movies, features = list(map(int,f.readline().split()))
#     for i in range(movies):
#         m_features = [None]*features
#         m_id = 0
#         counter = 0
#         for idx in range(features):
#             m_id, m_feature, m_value = f.readline().split()
#             m_features[idx] = m_value
#             counter += float(m_value)
#         if counter:
#             m_features = [m_id] + m_features
#             f_movies.write(','.join(m_features) + '\n')
#     f.close()
#     f_movies.close()
